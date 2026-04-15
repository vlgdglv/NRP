#!/usr/bin/env python3
"""
Position-wise conditional entropy heatmaps.

Reuses the same extraction / bucketing from conditional_entropy.py,
but maintains per-(r,c) counters to produce spatial maps of:
  H(X|Up), H(X|Left), H(X|Up,Left), LeftGain, UpGain

Usage:
    python spatial_analysis/conditional_entropy_heatmap.py \
        --data_dir /path/to/tokens \
        --dataset_name COCO \
        --model_type lumina \
        --num_samples 2000 \
        --top_m 256
"""
import sys, os, argparse, math
from pathlib import Path
from collections import Counter
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import TokenDataset
from spatial_analysis.conditional_entropy import (
    CONFIGS, extract_grid, build_top_m_map, remap_grid, OTHER,
)

# ---------------------------------------------------------------------------
# Per-position entropy from per-position counters
# ---------------------------------------------------------------------------

def entropy_bits(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def cond_entropy_bits(joint: Counter, marginal: Counter) -> float:
    """H(X|Y) from joint (x,y...) and marginal (y...) counters."""
    total = sum(joint.values())
    if total == 0:
        return float("nan")
    h = 0.0
    for key, c_xy in joint.items():
        if c_xy == 0:
            continue
        m_key = key[1:] if len(key) > 2 else key[1]
        c_y = marginal[m_key]
        if c_y == 0:
            continue
        p_xy = c_xy / total
        p_x_given_y = c_xy / c_y
        h -= p_xy * math.log2(p_x_given_y)
    return h


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(data, title, path, vmin=None, vmax=None, cmap="viridis"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, data.shape[1] * 0.18), max(4, data.shape[0] * 0.14)))
    im = ax.imshow(data, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="bits")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_side_by_side(data_l, data_r, title_l, title_r, path, cmap="viridis"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmin = np.nanmin([np.nanmin(data_l), np.nanmin(data_r)])
    vmax = np.nanmax([np.nanmax(data_l), np.nanmax(data_r)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, data_l.shape[1] * 0.3), max(4, data_l.shape[0] * 0.14)))
    im1 = ax1.imshow(data_l, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_xlabel("col"); ax1.set_ylabel("row"); ax1.set_title(title_l)
    fig.colorbar(im1, ax=ax1, label="bits")

    im2 = ax2.imshow(data_r, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_xlabel("col"); ax2.set_ylabel("row"); ax2.set_title(title_r)
    fig.colorbar(im2, ax=ax2, label="bits")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_quad(maps, titles, path, cmap="viridis"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(max(14, maps[0].shape[1] * 0.3), max(8, maps[0].shape[0] * 0.25)))
    for ax, data, title in zip(axes.flat, maps, titles):
        im = ax.imshow(data, aspect="auto", origin="upper", cmap=cmap)
        ax.set_xlabel("col"); ax.set_ylabel("row"); ax.set_title(title)
        fig.colorbar(im, ax=ax, label="bits")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Position-wise conditional entropy heatmaps")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--model_type", type=str, default="lumina", choices=["lumina", "janus"])
    p.add_argument("--grid_h", type=int, default=None)
    p.add_argument("--grid_w", type=int, default=None)
    p.add_argument("--top_m", type=int, default=256)
    p.add_argument("--exclude_eol", action="store_true", default=True)
    p.add_argument("--include_eol", action="store_true")
    p.add_argument("--output_dir", type=str, default="outputs/conditional_entropy/plots")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = CONFIGS[args.model_type]
    H = args.grid_h or cfg["H"]
    W = args.grid_w or cfg["W"]
    eoi = cfg["eoi"]
    exclude_eol = args.exclude_eol and not args.include_eol

    print(f"Model: {args.model_type}  Grid: {H}x{W}  top_m: {args.top_m}  exclude_eol: {exclude_eol}")

    # ---- load ----
    ds = TokenDataset(data_dir=args.data_dir, dataset_name=args.dataset_name, use_teacher=False)
    total = len(ds) if args.num_samples is None else min(args.num_samples, len(ds))
    print(f"Dataset: {len(ds)} samples, using {total}")

    # ================================================================
    # Pass 1: raw freq for top-M
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
        for v in grid.flat:
            raw_freq[int(v)] += 1
        grids.append(grid)

    M = min(args.top_m, len(raw_freq))
    mapping = build_top_m_map(raw_freq, M)
    print(f"Loaded {len(grids)} grids, vocab={len(raw_freq)}, top-{M}")

    eff_H, eff_W = grids[0].shape

    # ================================================================
    # Pass 2: per-position counters
    # ================================================================
    print("\nPass 2: per-position co-occurrence counts...")

    # For each (r,c): counters for x, joint(x,u), joint(x,l), joint(x,u,l), marginal u, l, (u,l)
    # Use flat arrays of Counters indexed by (r*eff_W + c)
    sz = eff_H * eff_W
    pos_count_x   = [Counter() for _ in range(sz)]
    pos_count_xu  = [Counter() for _ in range(sz)]  # (x, u)
    pos_count_xl  = [Counter() for _ in range(sz)]  # (x, l)
    pos_count_xul = [Counter() for _ in range(sz)]  # (x, u, l)
    pos_count_u   = [Counter() for _ in range(sz)]
    pos_count_l   = [Counter() for _ in range(sz)]
    pos_count_ul  = [Counter() for _ in range(sz)]  # (u, l)

    for grid in tqdm(grids, desc="Pass 2"):
        bg = remap_grid(grid, mapping)
        for r in range(eff_H):
            for c in range(eff_W):
                idx = r * eff_W + c
                x = int(bg[r, c])
                pos_count_x[idx][x] += 1

                has_up = r > 0
                has_left = c > 0

                if has_up:
                    u = int(bg[r - 1, c])
                    pos_count_xu[idx][(x, u)] += 1
                    pos_count_u[idx][u] += 1

                if has_left:
                    l = int(bg[r, c - 1])
                    pos_count_xl[idx][(x, l)] += 1
                    pos_count_l[idx][l] += 1

                if has_up and has_left:
                    u = int(bg[r - 1, c])
                    l = int(bg[r, c - 1])
                    pos_count_xul[idx][(x, u, l)] += 1
                    pos_count_ul[idx][(u, l)] += 1

    # ================================================================
    # Compute per-position entropies
    # ================================================================
    print("\nComputing per-position entropies...")

    map_Hx       = np.full((eff_H, eff_W), np.nan)
    map_Hx_up    = np.full((eff_H, eff_W), np.nan)
    map_Hx_left  = np.full((eff_H, eff_W), np.nan)
    map_Hx_upleft = np.full((eff_H, eff_W), np.nan)

    for r in range(eff_H):
        for c in range(eff_W):
            idx = r * eff_W + c
            map_Hx[r, c] = entropy_bits(pos_count_x[idx])
            if r > 0:
                map_Hx_up[r, c] = cond_entropy_bits(pos_count_xu[idx], pos_count_u[idx])
            if c > 0:
                map_Hx_left[r, c] = cond_entropy_bits(pos_count_xl[idx], pos_count_l[idx])
            if r > 0 and c > 0:
                map_Hx_upleft[r, c] = cond_entropy_bits(pos_count_xul[idx], pos_count_ul[idx])

    # Gain maps (only valid where both operands exist)
    left_gain = map_Hx_up - map_Hx_upleft      # H(X|Up) - H(X|Up,Left)
    up_gain   = map_Hx_left - map_Hx_upleft     # H(X|Left) - H(X|Up,Left)

    # ================================================================
    # Print summary stats
    # ================================================================
    def _stat(name, arr):
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return
        print(f"  {name:28s}  mean={valid.mean():.4f}  std={valid.std():.4f}  min={valid.min():.4f}  max={valid.max():.4f}")

    print("\n" + "=" * 70)
    print("POSITION-WISE ENTROPY SUMMARY  (bits)")
    print("=" * 70)
    _stat("H(X)", map_Hx)
    _stat("H(X|Up)", map_Hx_up)
    _stat("H(X|Left)", map_Hx_left)
    _stat("H(X|Up,Left)", map_Hx_upleft)
    _stat("LeftGain = H(X|Up)-H(X|Up,L)", left_gain)
    _stat("UpGain = H(X|Left)-H(X|Up,L)", up_gain)
    print("=" * 70)

    # ================================================================
    # Save plots
    # ================================================================
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_label = args.dataset_name or "default"
    prefix = f"{ds_label}_{args.model_type}_topm{M}"

    print("\nSaving heatmaps...")

    plot_heatmap(map_Hx_up,    f"H(X|Up)  [{ds_label}]",
                 out_dir / f"{prefix}_Hx_given_up.png")
    plot_heatmap(map_Hx_left,  f"H(X|Left)  [{ds_label}]",
                 out_dir / f"{prefix}_Hx_given_left.png")
    plot_heatmap(map_Hx_upleft, f"H(X|Up,Left)  [{ds_label}]",
                 out_dir / f"{prefix}_Hx_given_up_left.png")
    plot_heatmap(left_gain,    f"LeftGain = H(X|Up) - H(X|Up,Left)  [{ds_label}]",
                 out_dir / f"{prefix}_left_gain.png", cmap="hot")
    plot_heatmap(up_gain,      f"UpGain = H(X|Left) - H(X|Up,Left)  [{ds_label}]",
                 out_dir / f"{prefix}_up_gain.png", cmap="hot")

    # Side-by-side: H(X|Up) vs H(X|Left)
    plot_side_by_side(map_Hx_up, map_Hx_left,
                      f"H(X|Up)", f"H(X|Left)",
                      out_dir / f"{prefix}_up_vs_left.png")

    # Side-by-side: LeftGain vs UpGain
    plot_side_by_side(left_gain, up_gain,
                      f"LeftGain", f"UpGain",
                      out_dir / f"{prefix}_left_vs_up_gain.png", cmap="hot")

    # Quad overview
    plot_quad(
        [map_Hx_up, map_Hx_left, left_gain, up_gain],
        ["H(X|Up)", "H(X|Left)", "LeftGain", "UpGain"],
        out_dir / f"{prefix}_quad_overview.png",
    )

    # ================================================================
    # Save raw arrays
    # ================================================================
    npz_path = out_dir / f"{prefix}_arrays.npz"
    np.savez(
        npz_path,
        Hx=map_Hx,
        Hx_up=map_Hx_up,
        Hx_left=map_Hx_left,
        Hx_upleft=map_Hx_upleft,
        left_gain=left_gain,
        up_gain=up_gain,
    )
    print(f"  saved arrays: {npz_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
