from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.simulation.closed_loop import run_closed_loop


def default_config() -> Dict[str, float | int | tuple[float, float]]:
    return {
        "N": 15,
        "duration_min": 45.0,
        "servers": 3,
        "mu": 1.2,
        "base_rate": 3.0,
        "peak_rate": 10.0,
        "sigma": 18.0,
        "K0": 3.0,
        "Kmax": 15.0,
        "eta_dist": (0.0, 10.0),
        "mc_samples": 60,
        "burn_in": 0.2,
    }


def compute_stable_area(mask: np.ndarray) -> float:
    area = float(np.sum(mask.astype(np.float64), dtype=np.float64))
    return area


def validate_stable_area(area: float) -> None:
    assert area > 0.0


def run_experiment(
    seeds: int = 10,
    gamma_points: int = 11,
    lambda_points: int = 11,
    out_npz: Path = Path("results/boundary_scan.npz"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if seeds < 1:
        raise ValueError("seeds must be >= 1")

    cfg = default_config()
    gamma_grid = np.linspace(0.0, 1.0, gamma_points, dtype=np.float64)
    lambda_grid = np.linspace(0.0, 3.0, lambda_points, dtype=np.float64)

    tau_mean = np.zeros((gamma_points, lambda_points), dtype=np.float64)

    for i, gamma in enumerate(gamma_grid):
        for j, lam in enumerate(lambda_grid):
            vals = np.zeros(seeds, dtype=np.float64)
            for seed in range(seeds):
                series, _ = run_closed_loop(
                    config=cfg,
                    gamma=float(gamma),
                    lambda_=float(lam),
                    seed=7000 + seed,
                )
                arr = np.nan_to_num(np.asarray(series, dtype=np.float64), nan=0.0)
                vals[seed] = float(np.mean(arr, dtype=np.float64))
            tau_mean[i, j] = float(np.mean(vals, dtype=np.float64))

    threshold = float(np.quantile(tau_mean, 0.4))
    stable_mask = tau_mean <= threshold
    stable_area = compute_stable_area(stable_mask)
    validate_stable_area(stable_area)

    Path("results").mkdir(exist_ok=True)
    np.savez_compressed(
        out_npz,
        gamma_grid=gamma_grid,
        lambda_grid=lambda_grid,
        tau_mean=tau_mean,
        stable_mask=stable_mask.astype(np.int8),
        stable_area=np.asarray([stable_area], dtype=np.float64),
    )

    return gamma_grid, lambda_grid, tau_mean, stable_mask, stable_area


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--gamma_points", type=int, default=11)
    parser.add_argument("--lambda_points", type=int, default=11)
    args = parser.parse_args()
    run_experiment(
        seeds=args.seeds,
        gamma_points=args.gamma_points,
        lambda_points=args.lambda_points,
    )


# pytest assertion example

def test_boundary_scan_assertion_example() -> None:
    demo_mask = np.asarray([[1, 0], [0, 1]], dtype=np.int8)
    area = compute_stable_area(demo_mask)
    validate_stable_area(area)


if __name__ == "__main__":
    main()
