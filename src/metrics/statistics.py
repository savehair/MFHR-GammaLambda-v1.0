from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import stats


def paired_t_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Paired t-test for two matched samples."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)

    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays must be non-empty.")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape for paired t-test.")

    t_stat, p_value = stats.ttest_rel(x, y, nan_policy="omit")
    mean_diff = float(np.nanmean(x - y))

    return {
        "t_stat": float(np.nan_to_num(t_stat, nan=0.0)),
        "p_value": float(np.nan_to_num(p_value, nan=1.0)),
        "mean_diff": mean_diff,
    }


def bootstrap_ci(
    x: np.ndarray,
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for the sample mean."""
    data = np.asarray(x, dtype=np.float64).reshape(-1)
    if data.size == 0:
        raise ValueError("Input array must be non-empty.")
    if n_boot < 1000:
        raise ValueError("n_boot must be >= 1000.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1).")

    rng = np.random.default_rng(seed)
    n = data.size
    boot_means = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(sample, dtype=np.float64)

    alpha = 1.0 - confidence
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return lo, hi


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size using pooled standard deviation."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)

    if x.size < 2 or y.size < 2:
        raise ValueError("Each input must have at least 2 samples.")

    nx = x.size
    ny = y.size
    vx = np.var(x, ddof=1, dtype=np.float64)
    vy = np.var(y, ddof=1, dtype=np.float64)

    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled <= 0.0:
        return 0.0

    return float((np.mean(x, dtype=np.float64) - np.mean(y, dtype=np.float64)) / pooled)


def required_sample_size(
    delta: float,
    sigma: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Approximate per-group sample size for two-sided z-test."""
    if delta <= 0.0:
        raise ValueError("delta must be > 0.")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if not (0.0 < power < 1.0):
        raise ValueError("power must be in (0, 1).")

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) * sigma / delta) ** 2
    return int(np.ceil(n))


def t_test(a: np.ndarray, b: np.ndarray) -> float:
    """Backward-compatible independent two-sample p-value."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    _, p_val = stats.ttest_ind(x, y, nan_policy="omit")
    return float(np.nan_to_num(p_val, nan=1.0))


# pytest assertion examples

def test_paired_t_test_identical_arrays() -> None:
    a = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    b = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    result = paired_t_test(a, b)
    assert result["p_value"] > 0.99


def test_paired_t_test_different_arrays() -> None:
    a = np.asarray([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float64)
    b = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = paired_t_test(a, b)
    assert result["p_value"] < 0.05
