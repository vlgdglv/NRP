"""Tiny AR head for within-row coupling.

Architecture: small causal transformer over W positions of one row, conditioned on
backbone hiddens (additive). Operates at d_trunk so the caller can project outputs
through the existing trainable head (Janus: gen_head in modules_to_save).

Inputs/outputs work in trunk-dim space (no proj_in, no own vocab projection).
The caller is responsible for:
  - embedding previous-row tokens (e.g. via prepare_gen_img_embeds for Janus)
  - projecting the head's hidden output to vocab logits via the model's gen_head/lm_head

Inference uses an O(W²) step helper — no KV cache (W ≤ 64, cost is negligible).
"""
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
    """Within-row AR head, operates at d_trunk dim.

    Training: forward(h_row, prev_embeds) -> hidden [B, W, d_trunk]
        h_row:       [B, W, d_trunk]   backbone hiddens at the W predictor positions
        prev_embeds: [B, W, d_trunk]   embeddings of previous target-row tokens;
                                         position 0 is overwritten by learned BOS.
        Caller projects returned hidden -> vocab via gen_head/lm_head.

    Inference: step(h_row, prev_embeds_so_far) -> hidden [B, 1, d_trunk]
        h_row:                [B, W, d_trunk]
        prev_embeds_so_far:   [B, c+1, d_trunk]   (BOS at pos 0, then embeddings of
                                                    sampled tokens 0..c-1)
        Returns hidden at column c; caller projects -> logits, samples, then for the
        next step extends prev_embeds_so_far with the new token's embedding.
    """

    def __init__(
        self,
        d_trunk: int,
        n_layer: int,
        n_head: int,
        W: int,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.W = W
        self.d_trunk = d_trunk

        self.bos = nn.Parameter(torch.zeros(d_trunk))
        self.pos_emb = nn.Parameter(torch.zeros(W, d_trunk))
        self.layers = nn.ModuleList([_TinyBlock(d_trunk, n_head) for _ in range(n_layer)])
        self.norm = RMSNorm(d_trunk)

        std = initializer_range
        nn.init.normal_(self.bos, mean=0.0, std=std)
        nn.init.normal_(self.pos_emb, mean=0.0, std=std)
        for blk in self.layers:
            nn.init.normal_(blk.qkv.weight, mean=0.0, std=std)
            nn.init.normal_(blk.wo.weight, mean=0.0, std=std)
            nn.init.normal_(blk.ff_w1.weight, mean=0.0, std=std)
            nn.init.normal_(blk.ff_w2.weight, mean=0.0, std=std)

    def _assemble(self, h_row: torch.Tensor, prev_embeds: torch.Tensor) -> torch.Tensor:
        """h_row: [B, L, D]; prev_embeds: [B, L, D] (col 0 dummy, overwritten by BOS)."""
        L = prev_embeds.shape[1]
        prev = prev_embeds.clone()
        prev[:, 0] = self.bos.to(dtype=prev.dtype)
        return h_row + prev + self.pos_emb[:L].unsqueeze(0).to(dtype=h_row.dtype)

    def forward(self, h_row: torch.Tensor, prev_embeds: torch.Tensor) -> torch.Tensor:
        """Teacher-forced parallel forward (training)."""
        x = self._assemble(h_row, prev_embeds)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)  # [B, W, D]

    @torch.no_grad()
    def step(self, h_row: torch.Tensor, prev_embeds_so_far: torch.Tensor) -> torch.Tensor:
        """Inference step: returns hidden at the next column to sample, [B, 1, D].

        Re-runs the first L positions each call (O(W²) total per row). For W≤64 the
        cost is dwarfed by the backbone forward; not worth a KV cache.
        """
        L = prev_embeds_so_far.shape[1]
        x = self._assemble(h_row[:, :L], prev_embeds_so_far)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x[:, -1:])  # [B, 1, D]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def embed_target_row(base_model, target_row_ids: torch.Tensor) -> torch.Tensor:
    """Embed target-row token IDs through the SAME pathway the backbone uses at inference."""
    if hasattr(base_model, "prepare_gen_img_embeds"):
        return base_model.prepare_gen_img_embeds(target_row_ids)
    return base_model.get_input_embeddings()(target_row_ids)


def get_output_head(base_model):
    """Return (callable) that maps [..., D] hidden -> [..., V] logits."""
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
