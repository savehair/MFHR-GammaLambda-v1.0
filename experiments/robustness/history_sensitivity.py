from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def simulate_wait_series(
    seed: int,
    n_steps: int = 360,
    n_nodes: int = 15,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=np.float64)
    series = np.zeros((n_steps, n_nodes), dtype=np.float64)

    for j in range(n_nodes):
        phase = 0.1 * j
        periodic = 8.0 + 2.0 * np.sin(2.0 * np.pi * t / 60.0 + phase)
        noise = rng.normal(0.0, 0.15, size=n_steps)
        series[:, j] = periodic + noise

    return series


def predict_with_history(y: np.ndarray, history: int) -> np.ndarray:
    if history < 60:
        pred = y[history - 1 : -1]
    else:
        pred = y[history - 60 : -60]
    return np.asarray(pred, dtype=np.float64)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(err), dtype=np.float64))


def validate_results(df: pd.DataFrame) -> None:
    mean_map = df.groupby("history")["MAE"].mean().to_dict()
    assert float(mean_map[60]) <= float(mean_map[30])


def run_experiment(
    seeds: int = 10,
    histories: Iterable[int] = (30, 60, 120),
    out_csv: Path = Path("results/history_sensitivity.csv"),
) -> pd.DataFrame:
    if seeds < 1:
        raise ValueError("seeds must be >= 1")

    rows: List[Dict[str, float | int]] = []

    for seed in range(seeds):
        y = simulate_wait_series(seed=4000 + seed)
        for history in histories:
            y_true = np.asarray(y[history:], dtype=np.float64)
            y_pred = predict_with_history(y, history=history)
            mae_val = compute_mae(y_true, y_pred)
            rows.append(
                {
                    "history": int(history),
                    "seed": int(seed),
                    "MAE": float(mae_val),
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

def test_history_sensitivity_assertion_example() -> None:
    demo = pd.DataFrame(
        {
            "history": [30, 30, 60, 60, 120, 120],
            "seed": [0, 1, 0, 1, 0, 1],
            "MAE": [1.2, 1.1, 0.8, 0.9, 0.85, 0.9],
        }
    )
    validate_results(demo)


if __name__ == "__main__":
    main()
