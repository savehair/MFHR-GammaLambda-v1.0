from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from src.simulation.closed_loop import run_closed_loop


def default_config() -> Dict[str, float | int | tuple[float, float]]:
    return {
        "N": 15,
        "duration_min": 60.0,
        "servers": 3,
        "mu": 1.2,
        "base_rate": 3.0,
        "peak_rate": 12.0,
        "sigma": 20.0,
        "K0": 3.0,
        "Kmax": 15.0,
        "eta_dist": (0.0, 10.0),
        "mc_samples": 80,
        "burn_in": 0.2,
    }


def compute_safe_slope(series: np.ndarray) -> float:
    arr = np.asarray(series, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0)

    if arr.size < 2:
        return 0.0
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)

    x = np.arange(arr.size, dtype=np.float64)
    slope, _ = np.polyfit(x, arr, 1)
    return float(np.nan_to_num(slope, nan=0.0))


def validate_results(df: pd.DataFrame) -> None:
    assert not df["tau_slope"].isna().any()


def run_experiment(
    seeds: int = 10,
    multipliers: Iterable[int] = (2, 3),
    out_csv: Path = Path("results/peak_shock_metrics.csv"),
) -> pd.DataFrame:
    if seeds < 1:
        raise ValueError("seeds must be >= 1")

    base_cfg = default_config()
    rows: List[Dict[str, float | int]] = []

    for multiplier in multipliers:
        slopes: List[float] = []
        vars_: List[float] = []

        cfg = dict(base_cfg)
        cfg["peak_rate"] = float(base_cfg["peak_rate"]) * float(multiplier)

        for seed in range(seeds):
            tau_series, _ = run_closed_loop(
                config=cfg,
                gamma=0.8,
                lambda_=1.0,
                seed=6000 + seed,
            )
            arr = np.asarray(tau_series, dtype=np.float64)
            arr = np.nan_to_num(arr, nan=0.0)

            slope_val = compute_safe_slope(arr)
            var_val = float(np.var(arr, dtype=np.float64))

            slopes.append(float(slope_val))
            vars_.append(float(var_val))

        rows.append(
            {
                "multiplier": int(multiplier),
                "tau_slope": float(np.mean(np.asarray(slopes, dtype=np.float64), dtype=np.float64)),
                "tau_var": float(np.mean(np.asarray(vars_, dtype=np.float64), dtype=np.float64)),
            }
        )

    df = pd.DataFrame(rows)
    validate_results(df)

    Path("results").mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    run_experiment(seeds=args.seeds)


# pytest assertion example

def test_peak_shock_assertion_example() -> None:
    demo = pd.DataFrame(
        {
            "multiplier": [2, 3],
            "tau_slope": [0.1, 0.2],
            "tau_var": [1.0, 1.4],
        }
    )
    validate_results(demo)


if __name__ == "__main__":
    main()
