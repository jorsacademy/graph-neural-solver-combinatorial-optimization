from __future__ import annotations

from math import comb

import numpy as np


def paired_bootstrap_ci(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("paired arrays must be one-dimensional and have equal shape")
    if len(baseline) == 0:
        raise ValueError("paired arrays must not be empty")
    diff = candidate - baseline
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        draw = rng.integers(0, len(diff), size=len(diff))
        means[index] = float(np.mean(diff[draw]))
    alpha = 1.0 - confidence
    return {
        "mean_difference": float(np.mean(diff)),
        "ci_low": float(np.quantile(means, alpha / 2.0)),
        "ci_high": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "win_rate": float(np.mean(candidate < baseline)),
    }


def exact_sign_test(baseline: np.ndarray, candidate: np.ndarray) -> float:
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("paired arrays must be one-dimensional and have equal shape")
    diff = candidate - baseline
    nonzero = diff[np.abs(diff) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    wins = int(np.sum(nonzero < 0.0))
    k = min(wins, n - wins)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))
