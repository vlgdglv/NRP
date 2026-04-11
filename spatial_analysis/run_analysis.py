#!/usr/bin/env python3
"""
Main script for running spatial analysis on image token datasets.

Usage:
    python -m spatial_analysis.run_analysis \
        --data_dir /path/to/tokens \
        --dataset_name COCO \
        --model_type lumina \
        --num_samples 1000 \
        --output_dir outputs/spatial_analysis
"""
import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import TokenDataset
from spatial_analysis.compute_stats import SpatialAnalyzer, run_analysis
from spatial_analysis.visualization import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spatial analysis of image token grids",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to token dataset directory")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Dataset name (subdirectory of data_dir)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to analyze (None = all)")

    # Model/grid arguments
    parser.add_argument("--model_type", type=str, default="lumina",
                        choices=["lumina", "janus"],
                        help="Model type for token extraction")
    parser.add_argument("--grid_height", type=int, default=None,
                        help="Grid height (default: 48 for lumina, 24 for janus)")
    parser.add_argument("--grid_width", type=int, default=None,
                        help="Grid width including EOL (default: 49 for lumina, 24 for janus)")
    parser.add_argument("--exclude_eol", action="store_true", default=True,
                        help="Exclude EOL tokens from analysis")
    parser.add_argument("--include_eol", action="store_true",
                        help="Include EOL tokens in analysis")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs/spatial_analysis",
                        help="Output directory for results and plots")
    parser.add_argument("--save_raw", action="store_true",
                        help="Save raw statistics as numpy files")

    return parser.parse_args()


def get_default_grid_size(model_type: str):
    """Get default grid dimensions for model type."""
    if model_type == "lumina":
        return 48, 49  # H, W (W includes EOL)
    elif model_type == "janus":
        return 24, 24
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def print_summary(stats):
    """Print summary statistics to console."""
    print("\n" + "=" * 60)
    print("SPATIAL ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nDataset: {stats.num_samples} samples")
    print(f"Grid size: {stats.grid_height} x {stats.grid_width}")

    print("\n--- Token Equality Rates ---")
    print(f"  eq_up    (vertical):     {stats.eq_up_mean:.6f}")
    print(f"  eq_left  (horizontal):   {stats.eq_left_mean:.6f}")
    print(f"  eq_ul    (upper-left):   {stats.eq_ul_mean:.6f}")
    print(f"  eq_ur    (upper-right):  {stats.eq_ur_mean:.6f}")

    ratio = stats.eq_up_mean / (stats.eq_left_mean + 1e-10)
    print(f"\n  Vertical/Horizontal ratio: {ratio:.3f}x")

    h_runs = np.array(stats.run_lengths_horizontal)
    v_runs = np.array(stats.run_lengths_vertical)

    print("\n--- Run-Length Statistics ---")
    print(f"  Horizontal: mean={h_runs.mean():.2f}, median={np.median(h_runs):.1f}, "
          f"p90={np.percentile(h_runs, 90):.1f}, max={h_runs.max()}")
    print(f"  Vertical:   mean={v_runs.mean():.2f}, median={np.median(v_runs):.1f}, "
          f"p90={np.percentile(v_runs, 90):.1f}, max={v_runs.max()}")

    if stats.sim_up_values:
        print("\n--- Embedding Similarity (mean ± std) ---")
        print(f"  sim_up:    {np.mean(stats.sim_up_values):.4f} ± {np.std(stats.sim_up_values):.4f}")
        print(f"  sim_left:  {np.mean(stats.sim_left_values):.4f} ± {np.std(stats.sim_left_values):.4f}")
        print(f"  sim_ul:    {np.mean(stats.sim_ul_values):.4f} ± {np.std(stats.sim_ul_values):.4f}")
        print(f"  sim_ur:    {np.mean(stats.sim_ur_values):.4f} ± {np.std(stats.sim_ur_values):.4f}")

    print("\n" + "=" * 60)

    # Interpretation
    print("\nINTERPRETATION:")
    if ratio > 1.0:
        print(f"  -> Vertical dependency is {ratio:.1f}x STRONGER than horizontal")
        print("  -> Row-wise prediction may face challenges: tokens depend more on above than left")
    elif ratio < 1.0:
        print(f"  -> Horizontal dependency is {1/ratio:.1f}x STRONGER than vertical")
        print("  -> Row-wise prediction is statistically supported!")
    else:
        print("  -> Vertical and horizontal dependencies are roughly equal")

    print("=" * 60 + "\n")

def save_results(stats, output_dir: Path, save_raw: bool = False):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary as JSON
    summary = {
        "num_samples": stats.num_samples,
        "grid_height": stats.grid_height,
        "grid_width": stats.grid_width,
        "equality_stats": {
            "eq_up_mean": float(stats.eq_up_mean),
            "eq_left_mean": float(stats.eq_left_mean),
            "eq_ul_mean": float(stats.eq_ul_mean),
            "eq_ur_mean": float(stats.eq_ur_mean),
            "vertical_horizontal_ratio": float(stats.eq_up_mean / (stats.eq_left_mean + 1e-10)),
        },
        "run_length_stats": {
            "horizontal": {
                "mean": float(np.mean(stats.run_lengths_horizontal)),
                "median": float(np.median(stats.run_lengths_horizontal)),
                "p90": float(np.percentile(stats.run_lengths_horizontal, 90)),
                "max": int(np.max(stats.run_lengths_horizontal)),
            },
            "vertical": {
                "mean": float(np.mean(stats.run_lengths_vertical)),
                "median": float(np.median(stats.run_lengths_vertical)),
                "p90": float(np.percentile(stats.run_lengths_vertical, 90)),
                "max": int(np.max(stats.run_lengths_vertical)),
            },
        },
        "timestamp": datetime.now().isoformat(),
    }

    if stats.sim_up_values:
        summary["similarity_stats"] = {
            "sim_up": {"mean": float(np.mean(stats.sim_up_values)), "std": float(np.std(stats.sim_up_values))},
            "sim_left": {"mean": float(np.mean(stats.sim_left_values)), "std": float(np.std(stats.sim_left_values))},
            "sim_ul": {"mean": float(np.mean(stats.sim_ul_values)), "std": float(np.std(stats.sim_ul_values))},
            "sim_ur": {"mean": float(np.mean(stats.sim_ur_values)), "std": float(np.std(stats.sim_ur_values))},
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {output_dir / 'summary.json'}")

    if save_raw:
        np.save(output_dir / "eq_up_heatmap.npy", stats.eq_up_heatmap)
        np.save(output_dir / "eq_left_heatmap.npy", stats.eq_left_heatmap)
        np.save(output_dir / "run_lengths_horizontal.npy", np.array(stats.run_lengths_horizontal))
        np.save(output_dir / "run_lengths_vertical.npy", np.array(stats.run_lengths_vertical))
        print(f"Saved raw numpy files to {output_dir}")


def main():
    args = parse_args()

    # Determine grid size
    default_H, default_W = get_default_grid_size(args.model_type)
    H = args.grid_height if args.grid_height else default_H
    W = args.grid_width if args.grid_width else default_W

    exclude_eol = args.exclude_eol and not args.include_eol

    print(f"\n{'=' * 60}")
    print("SPATIAL ANALYSIS OF IMAGE TOKEN GRIDS")
    print(f"{'=' * 60}")
    print(f"Model type: {args.model_type}")
    print(f"Grid size: {H} x {W} (exclude_eol={exclude_eol})")
    print(f"Data dir: {args.data_dir}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Num samples: {args.num_samples if args.num_samples else 'all'}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'=' * 60}\n")

    # Load dataset
    print("Loading dataset...")
    dataset = TokenDataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        use_teacher=False,
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # Create analyzer
    analyzer = SpatialAnalyzer(
        H=H,
        W=W,
        model_type=args.model_type,
        exclude_eol=exclude_eol,
        embeddings=None,  # Skip embedding analysis for now
    )

    # Run analysis
    print("\nRunning spatial analysis...")
    stats = run_analysis(
        dataset=dataset,
        analyzer=analyzer,
        num_samples=args.num_samples,
        show_progress=True,
    )

    # Print summary
    print_summary(stats)

    # Save results
    output_path = Path(args.output_dir)
    save_results(stats, output_path, save_raw=args.save_raw)

    # Generate plots
    generate_all_plots(stats, args.output_dir)

    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
