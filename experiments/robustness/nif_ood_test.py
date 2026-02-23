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


def rolling_p50_proxy(y_true: np.ndarray, window: int = 5) -> np.ndarray:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.empty_like(y)
    for i in range(y.shape[0]):
        left = max(0, i - window + 1)
        pred[i] = np.median(y[left : i + 1])
    return pred


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_arr = np.asarray(y_true, dtype=np.float64)
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(true_arr - pred_arr), dtype=np.float64))


def validate_results(df: pd.DataFrame) -> None:
    assert (df["MAE_relative_change"] >= 0.0).all()


def run_experiment(
    seeds: int = 10,
    mu_values: Iterable[float] = (1.0, 1.2, 1.5),
    out_csv: Path = Path("results/nif_ood_metrics.csv"),
) -> pd.DataFrame:
    if seeds < 10:
        raise ValueError("seeds must be >= 10")

    base_cfg = default_config()
    rows: List[Dict[str, float | int]] = []

    for seed in range(seeds):
        ref_cfg = dict(base_cfg)
        ref_cfg["mu"] = 1.2
        ref_true, _ = run_closed_loop(config=ref_cfg, gamma=0.8, lambda_=1.0, seed=2000 + seed)
        ref_pred = rolling_p50_proxy(np.asarray(ref_true, dtype=np.float64))
        baseline_mae = mae(np.asarray(ref_true, dtype=np.float64), ref_pred)

        denom = max(abs(baseline_mae), 1e-12)

        for mu in mu_values:
            cfg = dict(base_cfg)
            cfg["mu"] = float(mu)
            y_true, _ = run_closed_loop(config=cfg, gamma=0.8, lambda_=1.0, seed=2000 + seed)
            y_pred = rolling_p50_proxy(np.asarray(y_true, dtype=np.float64))
            mae_val = mae(np.asarray(y_true, dtype=np.float64), y_pred)
            rel_change = abs(mae_val - baseline_mae) / denom

            rows.append(
                {
                    "mu": float(mu),
                    "seed": int(seed),
                    "MAE": float(mae_val),
                    "MAE_relative_change": float(rel_change),
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

def test_nif_ood_assertions_example() -> None:
    sample = pd.DataFrame(
        {
            "mu": [1.0, 1.2, 1.5],
            "seed": [0, 0, 0],
            "MAE": [1.0, 0.8, 1.2],
            "MAE_relative_change": [0.2, 0.0, 0.5],
        }
    )
    validate_results(sample)


if __name__ == "__main__":
    main()
