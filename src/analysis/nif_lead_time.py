from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np


def compute_peak_lead_time(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute peak lead time (minutes).

    Lead time is defined as: argmax(y_true) - argmax(y_pred) when prediction peaks earlier,
    otherwise 0.
    """
    true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if true_arr.size == 0 or pred_arr.size == 0:
        raise ValueError("y_true and y_pred must be non-empty.")
    if true_arr.shape != pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    true_safe = np.nan_to_num(true_arr, nan=-np.inf)
    pred_safe = np.nan_to_num(pred_arr, nan=-np.inf)

    true_peak_idx = int(np.argmax(true_safe))
    pred_peak_idx = int(np.argmax(pred_safe))

    return float(max(true_peak_idx - pred_peak_idx, 0))


def _parse_values(raw: Sequence[str]) -> np.ndarray:
    values = [float(x) for x in raw]
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--y_true", nargs="+", required=False)
    parser.add_argument("--y_pred", nargs="+", required=False)
    args = parser.parse_args()

    if args.y_true is None or args.y_pred is None:
        y_true = np.asarray([0, 1, 3, 8, 5, 2], dtype=np.float64)
        y_pred = np.asarray([0, 2, 6, 4, 2, 1], dtype=np.float64)
    else:
        y_true = _parse_values(args.y_true)
        y_pred = _parse_values(args.y_pred)

    lead_time = compute_peak_lead_time(y_true, y_pred)
    print(f"peak_lead_time={lead_time:.1f}")


# pytest assertion examples

def test_compute_peak_lead_time_returns_five() -> None:
    y_true = np.zeros(20, dtype=np.float64)
    y_pred = np.zeros(20, dtype=np.float64)
    y_true[12] = 10.0
    y_pred[7] = 9.0
    assert compute_peak_lead_time(y_true, y_pred) == 5.0


def test_compute_peak_lead_time_not_negative() -> None:
    y_true = np.asarray([0.0, 1.0, 3.0, 2.0], dtype=np.float64)
    y_pred = np.asarray([0.0, 4.0, 1.0, 0.0], dtype=np.float64)
    assert compute_peak_lead_time(y_true, y_pred) == 0.0


if __name__ == "__main__":
    main()
