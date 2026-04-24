"""Tiny AR head for within-row coupling.

Motivation: RowExpertModel emits W independent marginals per row from one parallel
backbone forward — joint factorizes into product, no intra-row dependence. The TinyAR
head adds a small causal AR over the W positions of the predicted row, conditioned on
the backbone's row hidden states + previously-generated tokens within the same row.
This breaks the factorization without adding a backbone forward.

Inference cost: 1 backbone parallel forward + W sequential mini-forwards through a
1-layer transformer (much smaller than backbone). Net ~10-20% over pure parallel.

The head outputs hidden states of base-model dim D; the caller projects via the base
output head (Janus: gen_head, Lumina: lm_head) so we share weights and avoid a 33-270M
new vocab projection.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyARHead(nn.Module):
    """1-layer (or N-layer) pre-norm transformer block, causal, KV-cache friendly.

    Forward inputs:
        backbone_hidden:  [N, W, D]  — backbone hidden states at the W predictor positions
                                       (i.e. row r's hidden states, used to predict row r+1)
        prev_token_embeds:[N, W, D]  — embedding of the previous *target-row* token at each
                                       step. Position 0 uses a learned BOS; position c>=1
                                       uses embed(target_token_{c-1}).

    Returns:
        hidden:           [N, W, D]  — caller projects to vocab via base output head.
    """

    def __init__(self, hidden_dim: int, num_layers: int = 1, num_heads: int = 8,
                 ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.bos = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.bos, std=0.02)

        # Combine backbone_hidden + prev_token_embed -> stream input
        self.token_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.layers = nn.ModuleList([
            _TinyBlock(hidden_dim, num_heads, ffn_mult, dropout)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def make_bos_emb(self, batch: int, dtype, device) -> torch.Tensor:
        """[batch, 1, D] BOS embedding for inference step 0."""
        return self.bos.expand(batch, 1, self.hidden_dim).to(dtype=dtype, device=device)

    def combine(self, backbone_hidden: torch.Tensor, prev_token_embeds: torch.Tensor) -> torch.Tensor:
        x = backbone_hidden + self.token_proj(prev_token_embeds)
        return self.input_norm(x)

    def forward(self, backbone_hidden: torch.Tensor, prev_token_embeds: torch.Tensor) -> torch.Tensor:
        """Parallel teacher-forced forward (training)."""
        x = self.combine(backbone_hidden, prev_token_embeds)
        N, W, D = x.shape
        # additive causal mask in float (huggingface-style); -inf above diagonal
        mask = torch.zeros(W, W, device=x.device, dtype=x.dtype)
        mask.masked_fill_(torch.triu(torch.ones(W, W, dtype=torch.bool, device=x.device), diagonal=1),
                          float("-inf"))
        for layer in self.layers:
            x = layer(x, attn_mask=mask)
        return self.out_norm(x)

    @torch.no_grad()
    def step(self, x_step: torch.Tensor, kv_cache: list) -> tuple:
        """Single inference step. x_step: [N, 1, D]. Returns (out [N,1,D], new kv_cache)."""
        new_cache = []
        h = x_step
        for li, layer in enumerate(self.layers):
            h, layer_kv = layer.step(h, kv_cache[li] if kv_cache else None)
            new_cache.append(layer_kv)
        return self.out_norm(h), new_cache


class _TinyBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        self.norm2 = nn.LayerNorm(dim)
        ffn_dim = ffn_mult * dim
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, dim, bias=False),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _attn(self, x, attn_mask=None, kv_cache=None):
        """x: [N, T, D]. If kv_cache provided, T should be 1 and we extend cache."""
        N, T, D = x.shape
        qkv = self.qkv(x).reshape(N, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [N, T, H, hd]
        # to [N, H, T, hd]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [N, H, T, S]
        if attn_mask is not None:
            attn = attn + attn_mask  # broadcast over [H]
        attn = attn.softmax(dim=-1).to(v.dtype)
        out = torch.matmul(attn, v)  # [N, H, T, hd]
        out = out.transpose(1, 2).contiguous().reshape(N, T, D)
        return self.proj(out), new_kv

    def forward(self, x, attn_mask=None):
        h, _ = self._attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

    def step(self, x, kv_cache):
        """Inference step with KV cache; no attn mask needed (single new query)."""
        h, new_kv = self._attn(self.norm1(x), attn_mask=None, kv_cache=kv_cache)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


# ============================================================================
# Helpers for embedding the previous-row tokens (matches backbone input pathway)
# ============================================================================

def embed_target_row(base_model, target_row_ids: torch.Tensor) -> torch.Tensor:
    """Embed target-row token IDs through the SAME pathway the backbone uses.

    Janus image tokens go through gen_aligner(gen_embed(.)).
    Lumina (Chameleon) uses unified vocab via get_input_embeddings.
    """
    if hasattr(base_model, "prepare_gen_img_embeds"):
        return base_model.prepare_gen_img_embeds(target_row_ids)
    return base_model.get_input_embeddings()(target_row_ids)


def get_output_head(base_model):
    """Return (callable) that maps [..., D] hidden -> [..., V] logits."""
    if hasattr(base_model, "gen_head"):
        return base_model.gen_head
    # Lumina / generic causal LM
    head = getattr(base_model, "lm_head", None)
    if head is None:
        head = base_model.get_output_embeddings()
    return head


def get_backbone_hidden_dim(base_model) -> int:
    """Best-effort lookup of the backbone hidden dim."""
    # Janus
    if hasattr(base_model, "language_model"):
        cfg = base_model.language_model.config
        return getattr(cfg, "hidden_size", None) or cfg.n_embd
    # Lumina/Chameleon
    cfg = base_model.config
    return getattr(cfg, "hidden_size", None) or cfg.n_embd
