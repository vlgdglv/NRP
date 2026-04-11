#!/usr/bin/env python3
"""
Conditional entropy analysis for image token grids.

Estimates H(X), H(X|Up), H(X|Left), H(X|Up,Left) from token co-occurrence
counts, using a top-M frequency bucket to avoid sparsity.

Usage:
    python spatial_analysis/conditional_entropy.py \
        --data_dir /path/to/tokens \
        --dataset_name COCO \
        --model_type lumina \
        --num_samples 2000 \
        --top_m 256
"""
import sys, os, json, argparse, math
from pathlib import Path
from collections import Counter
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import TokenDataset

# ---------------------------------------------------------------------------
# Token extraction (same logic as spatial_analysis, inlined for simplicity)
# ---------------------------------------------------------------------------

CONFIGS = {
    "lumina": {"H": 48, "W": 49, "eoi": 8196},
    "janus":  {"H": 24, "W": 24, "eoi": 151847},
}


def extract_grid(token_seq: torch.Tensor, H: int, W: int, eoi: int, model_type: str):
    """Return (H, W) numpy int32 grid of image tokens, or None."""
    img_length = H * W
    if model_type == "janus":
        # Janus: image tokens are the last H*W tokens
        L = token_seq.shape[-1]
        start = L - img_length
        if start < 0:
            return None
        return token_seq[start:L].numpy().astype(np.int32).reshape(H, W)
    else:
        # Lumina: image tokens end right before EOI marker
        positions = (token_seq == eoi).nonzero(as_tuple=False)
        if positions.numel() == 0:
            return None
        eoi_pos = int(positions[0].item())
        start = eoi_pos - img_length
        if start < 0:
            return None
        return token_seq[start:eoi_pos].numpy().astype(np.int32).reshape(H, W)


# ---------------------------------------------------------------------------
# Top-M bucketing
# ---------------------------------------------------------------------------

OTHER = -1  # sentinel for the "other" bucket


def build_top_m_map(count_x: Counter, M: int) -> dict:
    """Return {raw_token_id: bucketed_id} mapping top-M frequent to themselves,
    everything else to OTHER."""
    top_m_ids = {tok for tok, _ in count_x.most_common(M)}
    return {tok: (tok if tok in top_m_ids else OTHER) for tok in count_x}


def remap_grid(grid: np.ndarray, mapping: dict) -> np.ndarray:
    """Apply bucketing to every element. Unknown ids map to OTHER."""
    out = np.empty_like(grid)
    for idx in np.ndindex(grid.shape):
        out[idx] = mapping.get(int(grid[idx]), OTHER)
    return out


# ---------------------------------------------------------------------------
# Entropy helpers  (all use log2 → bits)
# ---------------------------------------------------------------------------

def entropy_from_counts(counts: Counter, total: int) -> float:
    """H = -sum p log2 p"""
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def conditional_entropy(joint: Counter, marginal: Counter, total: int) -> float:
    """H(X|Y) = -sum_{x,y} p(x,y) log2 p(x|y)
    joint keys are (x, y...), marginal keys are y... (matching suffix)."""
    h = 0.0
    for key, c_xy in joint.items():
        if c_xy == 0:
            continue
        # marginal key = everything except first element
        m_key = key[1:] if len(key) > 2 else key[1]
        c_y = marginal[m_key]
        if c_y == 0:
            continue
        p_xy = c_xy / total
        p_x_given_y = c_xy / c_y
        h -= p_xy * math.log2(p_x_given_y)
    return h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Conditional entropy of image token grids")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--model_type", type=str, default="lumina", choices=["lumina", "janus"])
    p.add_argument("--grid_h", type=int, default=None)
    p.add_argument("--grid_w", type=int, default=None)
    p.add_argument("--top_m", type=int, default=256)
    p.add_argument("--exclude_eol", action="store_true", default=True,
                   help="Drop the last column (EOL) before analysis")
    p.add_argument("--include_eol", action="store_true")
    p.add_argument("--output_dir", type=str, default="outputs/conditional_entropy")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = CONFIGS[args.model_type]
    H = args.grid_h or cfg["H"]
    W = args.grid_w or cfg["W"]
    eoi = cfg["eoi"]
    exclude_eol = args.exclude_eol and not args.include_eol

    print(f"Model: {args.model_type}  Grid: {H}x{W}  top_m: {args.top_m}  exclude_eol: {exclude_eol}")

    # ---- load dataset ----
    ds = TokenDataset(data_dir=args.data_dir, dataset_name=args.dataset_name, use_teacher=False)
    total = len(ds) if args.num_samples is None else min(args.num_samples, len(ds))
    print(f"Dataset: {len(ds)} samples, using {total}")

    # ================================================================
    # Pass 1: collect raw token frequencies to determine top-M bucket
    # ================================================================
    print("\nPass 1: counting raw token frequencies...")
    raw_freq = Counter()
    grids = []

    for i in tqdm(range(total), desc="Pass 1"):
        grid = extract_grid(ds[i]["input_ids"], H, W, eoi, args.model_type)
        if grid is None:
            continue
        if exclude_eol:
            grid = grid[:, :-1]
        # Only internal positions (r>0, c>0)
        interior = grid[1:, 1:]
        for v in interior.flat:
            raw_freq[int(v)] += 1
        grids.append(grid)

    vocab_size = len(raw_freq)
    total_interior = sum(raw_freq.values())
    M = min(args.top_m, vocab_size)

    # Build mapping
    mapping = build_top_m_map(raw_freq, M)
    top_m_ids = {tok for tok, mapped in mapping.items() if mapped != OTHER}
    top_m_coverage = sum(raw_freq[t] for t in top_m_ids) / total_interior

    print(f"Unique tokens (interior): {vocab_size}")
    print(f"Top-{M} coverage: {top_m_coverage:.4f} ({top_m_coverage*100:.1f}%)")

    # Print top-20
    print("\nTop-20 most frequent tokens:")
    for rank, (tok, cnt) in enumerate(raw_freq.most_common(20), 1):
        print(f"  {rank:3d}. token {tok:6d}  count={cnt:8d}  ({cnt/total_interior*100:.2f}%)")

    # ================================================================
    # Pass 2: collect bucketed co-occurrence counts
    # ================================================================
    print("\nPass 2: collecting co-occurrence counts...")

    count_x = Counter()
    count_u = Counter()
    count_l = Counter()
    count_ul = Counter()   # (u, l)
    count_xu = Counter()   # (x, u)
    count_xl = Counter()   # (x, l)
    count_xul = Counter()  # (x, u, l)

    n_valid = 0

    for grid in tqdm(grids, desc="Pass 2"):
        bg = remap_grid(grid, mapping)
        eff_H, eff_W = bg.shape

        for r in range(1, eff_H):
            for c in range(1, eff_W):
                x = int(bg[r, c])
                u = int(bg[r - 1, c])
                l = int(bg[r, c - 1])

                count_x[x] += 1
                count_u[u] += 1
                count_l[l] += 1
                count_ul[(u, l)] += 1
                count_xu[(x, u)] += 1
                count_xl[(x, l)] += 1
                count_xul[(x, u, l)] += 1

                n_valid += 1

    # ================================================================
    # Compute entropies  (all in bits, log2)
    # ================================================================
    print(f"\nValid positions: {n_valid}")

    H_X = entropy_from_counts(count_x, n_valid)
    H_X_given_Up = conditional_entropy(count_xu, count_u, n_valid)
    H_X_given_Left = conditional_entropy(count_xl, count_l, n_valid)
    H_X_given_Up_Left = conditional_entropy(count_xul, count_ul, n_valid)

    I_X_Up = H_X - H_X_given_Up
    I_X_Left = H_X - H_X_given_Left
    I_X_Up_Left = H_X - H_X_given_Up_Left

    gain_left_given_up = H_X_given_Up - H_X_given_Up_Left
    gain_up_given_left = H_X_given_Left - H_X_given_Up_Left

    # ================================================================
    # Print results
    # ================================================================
    print("\n" + "=" * 60)
    print("CONDITIONAL ENTROPY ANALYSIS  (log2 → bits)")
    print("=" * 60)
    print(f"  Samples:          {len(grids)}")
    print(f"  Valid positions:  {n_valid}")
    print(f"  top_m:            {M}")
    print(f"  Bucket coverage:  {top_m_coverage:.4f}")
    print()
    print(f"  H(X)              = {H_X:.4f} bits")
    print(f"  H(X | Up)         = {H_X_given_Up:.4f} bits")
    print(f"  H(X | Left)       = {H_X_given_Left:.4f} bits")
    print(f"  H(X | Up, Left)   = {H_X_given_Up_Left:.4f} bits")
    print()
    print(f"  I(X; Up)          = {I_X_Up:.4f} bits")
    print(f"  I(X; Left)        = {I_X_Left:.4f} bits")
    print(f"  I(X; Up, Left)    = {I_X_Up_Left:.4f} bits")
    print()
    print(f"  H(X|Up) - H(X|Up,Left)     = {gain_left_given_up:.4f} bits  (extra info from Left, given Up)")
    print(f"  H(X|Left) - H(X|Up,Left)   = {gain_up_given_left:.4f} bits  (extra info from Up, given Left)")
    print("=" * 60)

    # Interpretation
    print("\nINTERPRETATION:")
    if I_X_Up > I_X_Left:
        print(f"  Up provides MORE information ({I_X_Up:.4f}) than Left ({I_X_Left:.4f})")
    else:
        print(f"  Left provides MORE information ({I_X_Left:.4f}) than Up ({I_X_Up:.4f})")
    print(f"  Knowing both reduces uncertainty by {I_X_Up_Left:.4f} bits total")
    print(f"  Left adds {gain_left_given_up:.4f} bits beyond Up alone")
    print(f"  Up adds {gain_up_given_left:.4f} bits beyond Left alone")

    # ================================================================
    # Save JSON
    # ================================================================
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_label = args.dataset_name or "default"
    out_file = out_dir / f"{ds_label}_topm{M}.json"

    results = {
        "model_type": args.model_type,
        "dataset_name": ds_label,
        "num_samples": len(grids),
        "num_valid_positions": n_valid,
        "top_m": M,
        "top_m_coverage": round(top_m_coverage, 6),
        "grid_h": H,
        "grid_w": W,
        "exclude_eol": exclude_eol,
        "H_X": round(H_X, 6),
        "H_X_given_Up": round(H_X_given_Up, 6),
        "H_X_given_Left": round(H_X_given_Left, 6),
        "H_X_given_Up_Left": round(H_X_given_Up_Left, 6),
        "I_X_Up": round(I_X_Up, 6),
        "I_X_Left": round(I_X_Left, 6),
        "I_X_Up_Left": round(I_X_Up_Left, 6),
        "gain_left_given_up": round(gain_left_given_up, 6),
        "gain_up_given_left": round(gain_up_given_left, 6),
        "timestamp": datetime.now().isoformat(),
    }

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
