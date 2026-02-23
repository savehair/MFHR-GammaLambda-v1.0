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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    true_arr = np.asarray(y_true, dtype=np.float64)
    pred_arr = np.asarray(y_pred, dtype=np.float64)

    err = true_arr - pred_arr
    mae_val = float(np.mean(np.abs(err), dtype=np.float64))
    rmse_val = float(np.sqrt(np.mean(np.square(err), dtype=np.float64)))

    true_peak = int(np.argmax(np.nan_to_num(true_arr, nan=-np.inf)))
    pred_peak = int(np.argmax(np.nan_to_num(pred_arr, nan=-np.inf)))
    peak_shift = float(abs(true_peak - pred_peak))

    return {
        "MAE": mae_val,
        "RMSE": rmse_val,
        "peak_shift": peak_shift,
    }


def validate_results(df: pd.DataFrame, seeds: int) -> None:
    assert len(df) == 3 * seeds
    assert not df[["MAE", "RMSE", "peak_shift"]].isna().any().any()
    assert (df["peak_shift"] >= 0.0).all()


def run_experiment(
    seeds: int = 10,
    multipliers: Iterable[int] = (1, 2, 3),
    out_csv: Path = Path("results/nif_extreme_flow.csv"),
) -> pd.DataFrame:
    if seeds < 10:
        raise ValueError("seeds must be >= 10")

    base_cfg = default_config()
    rows: List[Dict[str, float | int]] = []

    for flow_multiplier in multipliers:
        cfg = dict(base_cfg)
        cfg["base_rate"] = float(base_cfg["base_rate"]) * float(flow_multiplier)

        for seed in range(seeds):
            y_true, _ = run_closed_loop(
                config=cfg,
                gamma=0.8,
                lambda_=1.0,
                seed=1000 + seed,
            )
            y_pred = rolling_p50_proxy(np.asarray(y_true, dtype=np.float64))
            metrics = compute_metrics(np.asarray(y_true, dtype=np.float64), y_pred)

            rows.append(
                {
                    "flow_multiplier": int(flow_multiplier),
                    "seed": int(seed),
                    "MAE": float(metrics["MAE"]),
                    "RMSE": float(metrics["RMSE"]),
                    "peak_shift": float(metrics["peak_shift"]),
                }
            )

    df = pd.DataFrame(rows)
    validate_results(df, seeds=seeds)

    Path("results").mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    run_experiment(seeds=args.seeds)


# pytest assertion example

def test_nif_extreme_flow_assertions_example() -> None:
    seeds = 10
    sample = pd.DataFrame(
        {
            "flow_multiplier": np.repeat([1, 2, 3], seeds),
            "seed": np.tile(np.arange(seeds), 3),
            "MAE": np.ones(3 * seeds, dtype=np.float64),
            "RMSE": np.ones(3 * seeds, dtype=np.float64),
            "peak_shift": np.zeros(3 * seeds, dtype=np.float64),
        }
    )
    validate_results(sample, seeds=seeds)


if __name__ == "__main__":
    main()
