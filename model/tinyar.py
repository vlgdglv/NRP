"""Tiny AR head for within-row coupling (LlamaGen-style).

Design choices (learned from LlamaGen's proven tinyAR head):
  - Head owns its *own* token embedding and *own* output projection; does NOT
    reuse the backbone's gen_head/gen_embed (sharing them creates a tug-of-war
    between the main CE loss and the tinyAR loss and produces mushy outputs).
  - Projects the trunk hidden into head-dim space (d_head) via a fresh linear.
  - Position embedding within the W-wide row (learned).
  - Simple additive input: cond + tok + pos. No norm on the sum.
  - Pre-norm transformer blocks, RMSNorm, SDPA with is_causal=True.
  - At inference use the O(W²) step_logits helper (no KV cache needed for W<64).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps).to(x.dtype)
        return x_norm * self.weight


class _TinyBlock(nn.Module):
    def __init__(self, dim: int, n_head: int):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.norm2 = RMSNorm(dim)
        self.ff_w1 = nn.Linear(dim, 4 * dim, bias=False)
        self.ff_w2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        h = self.norm1(x)
        B, L, _ = h.shape
        qkv = self.qkv(h).view(B, L, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        x = x + self.wo(out)
        x = x + self.ff_w2(F.gelu(self.ff_w1(self.norm2(x))))
        return x


class TinyARHead(nn.Module):
    """Within-row AR head, LlamaGen-style.

    __init__:
        d_trunk:    backbone hidden dim (input to proj_in)
        d_head:     head internal dim (typical 384–512 for ~16k vocab, or match trunk)
        n_layer:    number of tiny transformer blocks
        n_head:     attention heads
        vocab_size: image vocab (Janus: 16384; Lumina image range subset, see below)
        W:          row width (number of positions per row)

    Training forward: forward(h_row, tokens) -> logits [B, W, V]
      h_row:  [B, W, d_trunk] — backbone hidden states at row r's W positions
                                 (these are the predictors of row r+1's W tokens)
      tokens: [B, W] long     — GROUND TRUTH row r+1 tokens (teacher-forced).
                                 column c sees tokens[:, c-1]; column 0 sees BOS.

    Inference: step_logits(h_row, sampled_so_far) -> [B, V]
      h_row:  [B, W, d_trunk]
      sampled_so_far: [B, c] long — tokens already sampled in this row (0 <= c < W)
      returns logits for column c; user samples then appends.

    Lumina note: image tokens live in a subset of the unified vocab [4, 8195].
    You can either (a) pass vocab_size = full unified vocab and use all token IDs
    directly, or (b) remap image tokens to a compact [0, 16384) range in the data
    pipeline and pass vocab_size=16384. Option (a) is simpler; it wastes a bit
    of embedding/out capacity on non-image rows of the table but costs nothing
    at inference. This module is vocab-agnostic.
    """

    def __init__(
        self,
        d_trunk: int,
        d_head: int,
        n_layer: int,
        n_head: int,
        vocab_size: int,
        W: int,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.W = W
        self.d_trunk = d_trunk
        self.d_head = d_head
        self.vocab_size = vocab_size

        self.proj_in = nn.Linear(d_trunk, d_head, bias=False)
        self.tok_emb = nn.Embedding(vocab_size, d_head)
        self.bos = nn.Parameter(torch.zeros(d_head))
        self.pos_emb = nn.Parameter(torch.zeros(W, d_head))
        self.layers = nn.ModuleList([_TinyBlock(d_head, n_head) for _ in range(n_layer)])
        self.norm = RMSNorm(d_head)
        self.out = nn.Linear(d_head, vocab_size, bias=False)

        std = initializer_range
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=std)
        nn.init.normal_(self.bos, mean=0.0, std=std)
        nn.init.normal_(self.pos_emb, mean=0.0, std=std)
        nn.init.normal_(self.proj_in.weight, mean=0.0, std=std)
        nn.init.normal_(self.out.weight, mean=0.0, std=std)
        for blk in self.layers:
            nn.init.normal_(blk.qkv.weight, mean=0.0, std=std)
            nn.init.normal_(blk.wo.weight, mean=0.0, std=std)
            nn.init.normal_(blk.ff_w1.weight, mean=0.0, std=std)
            nn.init.normal_(blk.ff_w2.weight, mean=0.0, std=std)

    def _assemble_inputs(self, cond: torch.Tensor, prev_tokens: torch.Tensor) -> torch.Tensor:
        """cond: [B, L, d_head]; prev_tokens: [B, L] long, col 0 dummy (overwritten by BOS)."""
        B, L = prev_tokens.shape
        tok = self.tok_emb(prev_tokens).clone()
        tok[:, 0] = self.bos
        return cond + tok + self.pos_emb[:L].unsqueeze(0)

    def forward(self, h_row: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-forced parallel forward (training)."""
        B, W = tokens.shape
        prev = torch.zeros_like(tokens)
        prev[:, 1:] = tokens[:, :-1]
        cond = self.proj_in(h_row.to(self.proj_in.weight.dtype))
        x = self._assemble_inputs(cond, prev)
        for layer in self.layers:
            x = layer(x)
        return self.out(self.norm(x))  # [B, W, V]

    @torch.no_grad()
    def step_logits(self, h_row: torch.Tensor, sampled_so_far: torch.Tensor) -> torch.Tensor:
        """Logits at next column to sample. O(W²) per row; cheap for W<64."""
        B = h_row.shape[0]
        c = sampled_so_far.shape[1]
        L = c + 1
        device = h_row.device
        prev = torch.zeros(B, L, dtype=torch.long, device=device)
        if c > 0:
            prev[:, 1:] = sampled_so_far
        cond = self.proj_in(h_row[:, :L].to(self.proj_in.weight.dtype))
        x = self._assemble_inputs(cond, prev)
        for layer in self.layers:
            x = layer(x)
        return self.out(self.norm(x[:, -1]))  # [B, V]


# ----------------------------------------------------------------------------
# Helpers (kept for backwards-compat; tinyAR path no longer uses them, but
# other code in modeling_draft.py imports them).
# ----------------------------------------------------------------------------

def embed_target_row(base_model, target_row_ids: torch.Tensor) -> torch.Tensor:
    if hasattr(base_model, "prepare_gen_img_embeds"):
        return base_model.prepare_gen_img_embeds(target_row_ids)
    return base_model.get_input_embeddings()(target_row_ids)


def get_output_head(base_model):
    if hasattr(base_model, "gen_head"):
        return base_model.gen_head
    head = getattr(base_model, "lm_head", None)
    if head is None:
        head = base_model.get_output_embeddings()
    return head


def get_backbone_hidden_dim(base_model) -> int:
    if hasattr(base_model, "language_model"):
        cfg = base_model.language_model.config
        return getattr(cfg, "hidden_size", None) or cfg.n_embd
    cfg = base_model.config
    return getattr(cfg, "hidden_size", None) or cfg.n_embd


def get_image_vocab_size(base_model) -> int:
    """Janus: gen_head.out_features (16384). Lumina/Chameleon: full unified vocab."""
    head = get_output_head(base_model)
    return head.weight.shape[0]
