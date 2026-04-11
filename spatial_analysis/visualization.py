"""
Visualization functions for spatial analysis results.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from pathlib import Path
from typing import Optional

from .compute_stats import SpatialStats


def setup_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })


def plot_equality_bar_chart(stats: SpatialStats, output_path: Path):
    """
    Plot bar chart comparing eq_up, eq_left, eq_ul, eq_ur global means.
    """
    setup_style()

    labels = ['Up\n(vertical)', 'Left\n(horizontal)', 'Upper-Left\n(diagonal)', 'Upper-Right\n(diagonal)']
    values = [stats.eq_up_mean, stats.eq_left_mean, stats.eq_ul_mean, stats.eq_ur_mean]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Token Equality Rate')
    ax.set_title(f'Neighborhood Token Equality Statistics\n(n={stats.num_samples} samples, {stats.grid_height}x{stats.grid_width} grid)')
    ax.set_ylim(0, max(values) * 1.2)

    # Add horizontal line at mean
    mean_val = np.mean(values)
    ax.axhline(y=mean_val, color='gray', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.4f}')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path / 'equality_bar_chart.png')
    plt.savefig(output_path / 'equality_bar_chart.pdf')
    plt.close()
    print(f"Saved: {output_path / 'equality_bar_chart.png'}")

def plot_similarity_distribution(stats: SpatialStats, output_path: Path):
    """
    Plot distribution of embedding cosine similarities.
    """
    if not stats.sim_up_values:
        print("Skipping similarity plot: no embedding data available")
        return

    setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    data_dict = {
        'Up (vertical)': stats.sim_up_values,
        'Left (horizontal)': stats.sim_left_values,
        'Upper-Left (diag)': stats.sim_ul_values,
        'Upper-Right (diag)': stats.sim_ur_values,
    }

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

    for (label, data), color in zip(data_dict.items(), colors):
        if data:
            sns.kdeplot(data, ax=ax, label=f'{label} (μ={np.mean(data):.3f})', color=color, linewidth=2)

    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title(f'Embedding Similarity Distribution by Direction\n(n={stats.num_samples} samples)')
    ax.legend()
    ax.set_xlim(-0.5, 1.0)

    plt.tight_layout()
    plt.savefig(output_path / 'similarity_distribution.png')
    plt.savefig(output_path / 'similarity_distribution.pdf')
    plt.close()
    print(f"Saved: {output_path / 'similarity_distribution.png'}")


def plot_run_length_distribution(stats: SpatialStats, output_path: Path):
    """
    Plot run-length distributions for horizontal vs vertical.
    """
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    h_runs = np.array(stats.run_lengths_horizontal)
    v_runs = np.array(stats.run_lengths_vertical)

    # Left plot: overlayed histogram
    ax = axes[0]
    max_len = max(h_runs.max(), v_runs.max()) if len(h_runs) > 0 and len(v_runs) > 0 else 10
    bins = np.arange(1, min(max_len + 2, 30))

    ax.hist(h_runs, bins=bins, alpha=0.6, label=f'Horizontal (μ={h_runs.mean():.2f})', color='#e74c3c', edgecolor='black')
    ax.hist(v_runs, bins=bins, alpha=0.6, label=f'Vertical (μ={v_runs.mean():.2f})', color='#2ecc71', edgecolor='black')

    ax.set_xlabel('Run Length')
    ax.set_ylabel('Count')
    ax.set_title('Run-Length Distribution (Histogram)')
    ax.legend()
    ax.set_yscale('log')

    # Right plot: KDE
    ax = axes[1]
    if len(h_runs) > 1:
        sns.kdeplot(h_runs, ax=ax, label=f'Horizontal (μ={h_runs.mean():.2f})', color='#e74c3c', linewidth=2, clip=(1, None))
    if len(v_runs) > 1:
        sns.kdeplot(v_runs, ax=ax, label=f'Vertical (μ={v_runs.mean():.2f})', color='#2ecc71', linewidth=2, clip=(1, None))

    ax.set_xlabel('Run Length')
    ax.set_ylabel('Density')
    ax.set_title('Run-Length Distribution (KDE)')
    ax.legend()
    ax.set_xlim(1, 15)

    plt.suptitle(f'Token Run-Length Analysis (n={stats.num_samples} samples)', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'run_length_distribution.png')
    plt.savefig(output_path / 'run_length_distribution.pdf')
    plt.close()
    print(f"Saved: {output_path / 'run_length_distribution.png'}")


def plot_equality_heatmaps(stats: SpatialStats, output_path: Path):
    """
    Plot heatmaps of position-wise equality rates.
    """
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    heatmaps = [
        (stats.eq_up_heatmap, 'eq_up (Vertical)', axes[0, 0]),
        (stats.eq_left_heatmap, 'eq_left (Horizontal)', axes[0, 1]),
        (stats.eq_ul_heatmap, 'eq_ul (Upper-Left Diag)', axes[1, 0]),
        (stats.eq_ur_heatmap, 'eq_ur (Upper-Right Diag)', axes[1, 1]),
    ]

    for hmap, title, ax in heatmaps:
        if hmap is not None:
            # Subsample if too large
            H, W = hmap.shape
            step_h = max(1, H // 48)
            step_w = max(1, W // 48)
            hmap_sub = hmap[::step_h, ::step_w]

            im = ax.imshow(hmap_sub, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.3)
            ax.set_title(f'{title}\n(mean={hmap.mean():.4f})')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            plt.colorbar(im, ax=ax, label='Equality Rate')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)

    plt.suptitle(f'Position-wise Token Equality Heatmaps\n(n={stats.num_samples} samples)', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'equality_heatmaps.png')
    plt.savefig(output_path / 'equality_heatmaps.pdf')
    plt.close()
    print(f"Saved: {output_path / 'equality_heatmaps.png'}")


def plot_summary_comparison(stats: SpatialStats, output_path: Path):
    """
    Plot summary comparison: vertical vs horizontal dependency strength.
    """
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Equality comparison
    ax = axes[0]
    categories = ['Token Equality Rate']
    vertical = [stats.eq_up_mean]
    horizontal = [stats.eq_left_mean]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, vertical, width, label='Vertical (Up)', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, horizontal, width, label='Horizontal (Left)', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Rate')
    ax.set_title('Vertical vs Horizontal Dependency')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Add ratio annotation
    ratio = stats.eq_up_mean / (stats.eq_left_mean + 1e-10)
    ax.annotate(f'V/H Ratio: {ratio:.2f}x', xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right: Run length comparison
    ax = axes[1]
    h_runs = np.array(stats.run_lengths_horizontal)
    v_runs = np.array(stats.run_lengths_vertical)

    metrics = ['Mean', 'Median', 'P90']
    v_vals = [v_runs.mean(), np.median(v_runs), np.percentile(v_runs, 90)]
    h_vals = [h_runs.mean(), np.median(h_runs), np.percentile(h_runs, 90)]

    x = np.arange(len(metrics))
    bars1 = ax.bar(x - width/2, v_vals, width, label='Vertical', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, h_vals, width, label='Horizontal', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Run Length')
    ax.set_title('Run-Length Statistics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.suptitle(f'Spatial Dependency Summary (n={stats.num_samples} samples)', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'summary_comparison.png')
    plt.savefig(output_path / 'summary_comparison.pdf')
    plt.close()
    print(f"Saved: {output_path / 'summary_comparison.png'}")


def generate_all_plots(stats: SpatialStats, output_dir: str):
    """Generate all visualization plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots in {output_path}...")

    plot_equality_bar_chart(stats, output_path)
    plot_run_length_distribution(stats, output_path)
    plot_equality_heatmaps(stats, output_path)
    plot_summary_comparison(stats, output_path)

    # Only plot similarity if data available
    if stats.sim_up_values:
        plot_similarity_distribution(stats, output_path)

    print(f"\nAll plots saved to {output_path}")
