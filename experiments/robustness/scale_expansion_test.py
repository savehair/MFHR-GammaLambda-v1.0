from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from src.simulation.closed_loop import run_closed_loop


def build_config(n_stores: int) -> Dict[str, float | int | tuple[float, float]]:
    servers = max(1, int(round(n_stores / 5.0)))
    return {
        "N": int(n_stores),
        "duration_min": 60.0,
        "servers": int(servers),
        "mu": 1.2,
        "base_rate": float(1.0 * servers),
        "peak_rate": float(4.0 * servers),
        "sigma": 20.0,
        "K0": 3.0,
        "Kmax": 15.0,
        "eta_dist": (0.0, 10.0),
        "mc_samples": 60,
        "burn_in": 0.2,
    }


def validate_results(df: pd.DataFrame) -> None:
    baseline = float(df.loc[df["N"] == df["N"].min(), "tau_mean"].iloc[0])
    assert (df["tau_mean"] <= (2.0 * baseline)).all()


def run_experiment(
    seeds: int = 10,
    n_values: Iterable[int] = (5, 15, 30, 50),
    out_csv: Path = Path("results/scale_expansion.csv"),
) -> pd.DataFrame:
    if seeds < 1:
        raise ValueError("seeds must be >= 1")

    rows: List[Dict[str, float | int]] = []

    for n_stores in n_values:
        tau_means: List[float] = []
        tau_vars: List[float] = []
        fairness_vals: List[float] = []

        cfg = build_config(n_stores)
        for seed in range(seeds):
            tau_series, fairness = run_closed_loop(
                config=cfg,
                gamma=0.8,
                lambda_=1.0,
                seed=5000 + seed,
            )
            tau_arr = np.asarray(tau_series, dtype=np.float64)
            tau_arr = np.nan_to_num(tau_arr, nan=0.0)

            tau_means.append(float(np.mean(tau_arr, dtype=np.float64)))
            tau_vars.append(float(np.var(tau_arr, dtype=np.float64)))
            fairness_vals.append(float(np.nan_to_num(fairness, nan=1.0)))

        rows.append(
            {
                "N": int(n_stores),
                "tau_mean": float(np.mean(np.asarray(tau_means, dtype=np.float64), dtype=np.float64)),
                "tau_var": float(np.mean(np.asarray(tau_vars, dtype=np.float64), dtype=np.float64)),
                "Jain_index": float(np.mean(np.asarray(fairness_vals, dtype=np.float64), dtype=np.float64)),
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

def test_scale_expansion_assertion_example() -> None:
    demo = pd.DataFrame(
        {
            "N": [5, 15, 30, 50],
            "tau_mean": [1.0, 1.4, 1.8, 1.9],
            "tau_var": [0.1, 0.2, 0.3, 0.35],
            "Jain_index": [0.9, 0.88, 0.86, 0.84],
        }
    )
    validate_results(demo)


if __name__ == "__main__":
    main()
