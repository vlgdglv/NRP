"""
Core statistics computation for spatial analysis of image token grids.
"""
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm


@dataclass
class SpatialStats:
    """Container for spatial statistics results."""
    # Global equality stats
    eq_up_mean: float = 0.0
    eq_left_mean: float = 0.0
    eq_ul_mean: float = 0.0
    eq_ur_mean: float = 0.0

    # Position-wise heatmaps (H x W)
    eq_up_heatmap: Optional[np.ndarray] = None
    eq_left_heatmap: Optional[np.ndarray] = None
    eq_ul_heatmap: Optional[np.ndarray] = None
    eq_ur_heatmap: Optional[np.ndarray] = None

    # Counts for averaging
    eq_up_counts: Optional[np.ndarray] = None
    eq_left_counts: Optional[np.ndarray] = None

    # Embedding similarity stats (if available)
    sim_up_values: list = field(default_factory=list)
    sim_left_values: list = field(default_factory=list)
    sim_ul_values: list = field(default_factory=list)
    sim_ur_values: list = field(default_factory=list)

    # Run-length distributions
    run_lengths_horizontal: list = field(default_factory=list)
    run_lengths_vertical: list = field(default_factory=list)

    # Metadata
    num_samples: int = 0
    grid_height: int = 0
    grid_width: int = 0


def extract_image_tokens_lumina(token_seq: torch.Tensor, H: int = 48, W: int = 49,
                                 eoi_token_id: int = 8196) -> Optional[torch.Tensor]:
    """
    Extract image tokens from Lumina token sequence and reshape to grid.

    Returns:
        Tensor of shape (H, W) containing image tokens, or None if extraction fails.
    """
    img_length = H * W

    # Find EOI position
    eoi_positions = (token_seq == eoi_token_id).nonzero(as_tuple=False)
    if eoi_positions.numel() == 0:
        return None

    eoi_pos = int(eoi_positions[0].item())
    img_start = eoi_pos - img_length

    if img_start < 0:
        return None

    img_tokens = token_seq[img_start:eoi_pos]
    return img_tokens.reshape(H, W)


def extract_image_tokens_janus(token_seq: torch.Tensor, H: int = 24, W: int = 24,
                                eoi_token_id: int = 151847) -> Optional[torch.Tensor]:
    """Extract image tokens from Janus token sequence.
    Janus stores image tokens as the last H*W tokens in the sequence."""
    img_length = H * W
    L = token_seq.shape[-1]
    img_start = L - img_length

    if img_start < 0:
        return None

    img_tokens = token_seq[img_start:L]
    return img_tokens.reshape(H, W)

def compute_equality_stats(grid: np.ndarray) -> dict:
    """
    Compute neighborhood equality statistics for a single image grid.

    Args:
        grid: numpy array of shape (H, W) containing token ids

    Returns:
        Dictionary with equality counts and valid position counts
    """
    H, W = grid.shape

    # Initialize accumulators
    eq_up = np.zeros((H, W), dtype=np.float32)
    eq_left = np.zeros((H, W), dtype=np.float32)
    eq_ul = np.zeros((H, W), dtype=np.float32)
    eq_ur = np.zeros((H, W), dtype=np.float32)

    valid_up = np.zeros((H, W), dtype=np.float32)
    valid_left = np.zeros((H, W), dtype=np.float32)
    valid_ul = np.zeros((H, W), dtype=np.float32)
    valid_ur = np.zeros((H, W), dtype=np.float32)

    for r in range(H):
        for c in range(W):
            # eq_up: compare with (r-1, c)
            if r > 0:
                eq_up[r, c] = float(grid[r, c] == grid[r - 1, c])
                valid_up[r, c] = 1.0

            # eq_left: compare with (r, c-1)
            if c > 0:
                eq_left[r, c] = float(grid[r, c] == grid[r, c - 1])
                valid_left[r, c] = 1.0

            # eq_ul: compare with (r-1, c-1)
            if r > 0 and c > 0:
                eq_ul[r, c] = float(grid[r, c] == grid[r - 1, c - 1])
                valid_ul[r, c] = 1.0

            # eq_ur: compare with (r-1, c+1)
            if r > 0 and c < W - 1:
                eq_ur[r, c] = float(grid[r, c] == grid[r - 1, c + 1])
                valid_ur[r, c] = 1.0

    return {
        "eq_up": eq_up, "valid_up": valid_up,
        "eq_left": eq_left, "valid_left": valid_left,
        "eq_ul": eq_ul, "valid_ul": valid_ul,
        "eq_ur": eq_ur, "valid_ur": valid_ur,
    }


def compute_run_lengths(grid: np.ndarray) -> dict:
    """
    Compute run-length distributions for horizontal and vertical directions.

    Args:
        grid: numpy array of shape (H, W)

    Returns:
        Dictionary with horizontal and vertical run lengths
    """
    H, W = grid.shape
    horizontal_runs = []
    vertical_runs = []

    # Horizontal runs (along each row)
    for r in range(H):
        run_len = 1
        for c in range(1, W):
            if grid[r, c] == grid[r, c - 1]:
                run_len += 1
            else:
                horizontal_runs.append(run_len)
                run_len = 1
        horizontal_runs.append(run_len)  # Last run in row

    # Vertical runs (along each column)
    for c in range(W):
        run_len = 1
        for r in range(1, H):
            if grid[r, c] == grid[r - 1, c]:
                run_len += 1
            else:
                vertical_runs.append(run_len)
                run_len = 1
        vertical_runs.append(run_len)  # Last run in column

    return {
        "horizontal": horizontal_runs,
        "vertical": vertical_runs,
    }


def compute_embedding_similarity(grid: np.ndarray, embeddings: np.ndarray) -> dict:
    """
    Compute cosine similarity between neighboring tokens using embeddings.

    Args:
        grid: numpy array of shape (H, W) containing token ids
        embeddings: numpy array of shape (vocab_size, embed_dim)

    Returns:
        Dictionary with similarity values for each direction
    """
    H, W = grid.shape

    sim_up = []
    sim_left = []
    sim_ul = []
    sim_ur = []

    def cosine_sim(v1, v2):
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    for r in range(H):
        for c in range(W):
            e_curr = embeddings[grid[r, c]]

            if r > 0:
                e_up = embeddings[grid[r - 1, c]]
                sim_up.append(cosine_sim(e_curr, e_up))

            if c > 0:
                e_left = embeddings[grid[r, c - 1]]
                sim_left.append(cosine_sim(e_curr, e_left))

            if r > 0 and c > 0:
                e_ul = embeddings[grid[r - 1, c - 1]]
                sim_ul.append(cosine_sim(e_curr, e_ul))

            if r > 0 and c < W - 1:
                e_ur = embeddings[grid[r - 1, c + 1]]
                sim_ur.append(cosine_sim(e_curr, e_ur))

    return {
        "sim_up": sim_up,
        "sim_left": sim_left,
        "sim_ul": sim_ul,
        "sim_ur": sim_ur,
    }

class SpatialAnalyzer:
    """
    Main class for computing spatial statistics over a dataset.
    """

    def __init__(self, H: int, W: int, model_type: str = "lumina",
                 exclude_eol: bool = True, embeddings: Optional[np.ndarray] = None):
        """
        Args:
            H: Grid height
            W: Grid width (including EOL if present)
            model_type: "lumina" or "janus"
            exclude_eol: If True, exclude the last column (EOL tokens) from analysis
            embeddings: Optional embedding matrix for similarity computation
        """
        self.H = H
        self.W = W
        self.model_type = model_type
        self.exclude_eol = exclude_eol
        self.embeddings = embeddings

        # Effective dimensions for analysis
        self.eff_W = W - 1 if exclude_eol else W
        self.eff_H = H

        # Token config
        if model_type == "lumina":
            self.eoi_token_id = 8196
            self.eol_token_id = 8803
        elif model_type == "janus":
            self.eoi_token_id = 151847
            self.eol_token_id = 151846
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Accumulators
        self._reset_accumulators()

    def _reset_accumulators(self):
        """Reset all accumulators for fresh analysis."""
        H, W = self.eff_H, self.eff_W

        self.sum_eq_up = np.zeros((H, W), dtype=np.float64)
        self.sum_eq_left = np.zeros((H, W), dtype=np.float64)
        self.sum_eq_ul = np.zeros((H, W), dtype=np.float64)
        self.sum_eq_ur = np.zeros((H, W), dtype=np.float64)

        self.count_up = np.zeros((H, W), dtype=np.float64)
        self.count_left = np.zeros((H, W), dtype=np.float64)
        self.count_ul = np.zeros((H, W), dtype=np.float64)
        self.count_ur = np.zeros((H, W), dtype=np.float64)

        self.all_run_horizontal = []
        self.all_run_vertical = []

        self.all_sim_up = []
        self.all_sim_left = []
        self.all_sim_ul = []
        self.all_sim_ur = []

        self.num_samples = 0

    def process_sample(self, token_seq: torch.Tensor) -> bool:
        """
        Process a single token sequence.

        Returns:
            True if processing succeeded, False otherwise
        """
        # Extract image grid
        if self.model_type == "lumina":
            grid = extract_image_tokens_lumina(
                token_seq, H=self.H, W=self.W, eoi_token_id=self.eoi_token_id
            )
        else:
            grid = extract_image_tokens_janus(
                token_seq, H=self.H, W=self.W, eoi_token_id=self.eoi_token_id
            )

        if grid is None:
            return False

        grid_np = grid.numpy()

        # Exclude EOL column if requested
        if self.exclude_eol:
            grid_np = grid_np[:, :-1]

        # Compute equality stats
        eq_stats = compute_equality_stats(grid_np)

        self.sum_eq_up += eq_stats["eq_up"]
        self.sum_eq_left += eq_stats["eq_left"]
        self.sum_eq_ul += eq_stats["eq_ul"]
        self.sum_eq_ur += eq_stats["eq_ur"]

        self.count_up += eq_stats["valid_up"]
        self.count_left += eq_stats["valid_left"]
        self.count_ul += eq_stats["valid_ul"]
        self.count_ur += eq_stats["valid_ur"]

        # Compute run lengths
        runs = compute_run_lengths(grid_np)
        self.all_run_horizontal.extend(runs["horizontal"])
        self.all_run_vertical.extend(runs["vertical"])

        # Compute embedding similarity if available
        if self.embeddings is not None:
            sims = compute_embedding_similarity(grid_np, self.embeddings)
            self.all_sim_up.extend(sims["sim_up"])
            self.all_sim_left.extend(sims["sim_left"])
            self.all_sim_ul.extend(sims["sim_ul"])
            self.all_sim_ur.extend(sims["sim_ur"])

        self.num_samples += 1
        return True

    def get_results(self) -> SpatialStats:
        """Compute final statistics and return results."""
        stats = SpatialStats()
        stats.num_samples = self.num_samples
        stats.grid_height = self.eff_H
        stats.grid_width = self.eff_W

        # Avoid division by zero
        eps = 1e-10

        # Position-wise heatmaps
        stats.eq_up_heatmap = self.sum_eq_up / (self.count_up + eps)
        stats.eq_left_heatmap = self.sum_eq_left / (self.count_left + eps)
        stats.eq_ul_heatmap = self.sum_eq_ul / (self.count_ul + eps)
        stats.eq_ur_heatmap = self.sum_eq_ur / (self.count_ur + eps)

        stats.eq_up_counts = self.count_up
        stats.eq_left_counts = self.count_left

        # Global means
        stats.eq_up_mean = self.sum_eq_up.sum() / (self.count_up.sum() + eps)
        stats.eq_left_mean = self.sum_eq_left.sum() / (self.count_left.sum() + eps)
        stats.eq_ul_mean = self.sum_eq_ul.sum() / (self.count_ul.sum() + eps)
        stats.eq_ur_mean = self.sum_eq_ur.sum() / (self.count_ur.sum() + eps)

        # Run lengths
        stats.run_lengths_horizontal = self.all_run_horizontal
        stats.run_lengths_vertical = self.all_run_vertical

        # Embedding similarities
        stats.sim_up_values = self.all_sim_up
        stats.sim_left_values = self.all_sim_left
        stats.sim_ul_values = self.all_sim_ul
        stats.sim_ur_values = self.all_sim_ur

        return stats


def run_analysis(dataset, analyzer: SpatialAnalyzer, num_samples: Optional[int] = None,
                 show_progress: bool = True) -> SpatialStats:
    """
    Run spatial analysis over a dataset.

    Args:
        dataset: Dataset object with __getitem__ returning {"input_ids": tensor}
        analyzer: SpatialAnalyzer instance
        num_samples: Max samples to process (None = all)
        show_progress: Show tqdm progress bar

    Returns:
        SpatialStats with computed statistics
    """
    total = len(dataset) if num_samples is None else min(num_samples, len(dataset))

    iterator = range(total)
    if show_progress:
        iterator = tqdm(iterator, desc="Analyzing samples")

    failed = 0
    for i in iterator:
        sample = dataset[i]
        token_seq = sample["input_ids"]
        if not analyzer.process_sample(token_seq):
            failed += 1

    if failed > 0:
        print(f"Warning: {failed}/{total} samples failed to process")

    return analyzer.get_results()
