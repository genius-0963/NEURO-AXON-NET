"""
Count-Stratified Weighted Sampler.

Balances rare-count examples across ten count bins using inverse frequency weighting,
avoiding dataset duplication into validation/test splits.
"""

from __future__ import annotations

from typing import List, Sequence
import numpy as np


class CountStratifiedSampler:
    """
    Computes sample probabilities to balance 10 count bins uniformly.
    """
    def __init__(self, counts: Sequence[float], num_bins: int = 10):
        self.counts = np.asarray(counts, dtype=np.float64)
        self.num_bins = num_bins
        self.weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        if len(self.counts) == 0:
            return np.array([])

        min_c = float(np.min(self.counts))
        max_c = float(np.max(self.counts))

        if min_c == max_c:
            return np.ones(len(self.counts)) / float(len(self.counts))

        bins = np.linspace(min_c, max_c + 1e-5, self.num_bins + 1)
        bin_indices = np.digitize(self.counts, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        bin_counts = np.bincount(bin_indices, minlength=self.num_bins)
        # Avoid division by zero
        bin_counts = np.maximum(bin_counts, 1)

        # Inverse frequency weighting
        sample_weights = 1.0 / bin_counts[bin_indices]
        sample_probs = sample_weights / np.sum(sample_weights)
        return sample_probs

    def sample_indices(self, n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.choice(len(self.counts), size=n_samples, p=self.weights, replace=True)
