from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.simulation.closed_loop import run_closed_loop

def compute_stats(tau_series: np.ndarray, burn_in_frac: float) -> tuple[float, float, float]:
    burn = int(len(tau_series) * burn_in_frac)
    x = tau_series[burn:]
    t = np.arange(len(x), dtype=float)
    slope, _ = np.polyfit(t, x, 1)
    return float(np.mean(x)), float(np.var(x)), float(slope)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--gamma_points", type=int, default=9)
    ap.add_argument("--lambda_points", type=int, default=9)
    ap.add_argument("--lambda_max", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--burn_in", type=float, default=0.2)
    args = ap.parse_args()

    config = {
        "N": 15,
        "duration_min": 120.0,  # 真实 120 分钟
        "servers": 3,
        "mu": 1.2,  # users/min
        "base_rate": 3.0,  # NHPP base
        "peak_rate": 12.0,  # NHPP peak
        "sigma": 20.0,
        "K0": 3.0,
        "Kmax": 15.0,
        "eta_dist": (0.0, 10.0),  # travel ETA in minutes, uniform
        "mc_samples": 60, #临时改60
        "burn_in": 0.2
    }

    out_dir = Path(args.out_dir)
    (out_dir / "raw_runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    gamma_grid = np.linspace(0.0, 1.0, args.gamma_points)
    lambda_grid = np.linspace(0.0, args.lambda_max, args.lambda_points)

    tau_mean = np.zeros((len(gamma_grid), len(lambda_grid)), dtype=float)
    tau_var = np.zeros_like(tau_mean)
    tau_slope = np.zeros_like(tau_mean)

    for i, gamma in enumerate(gamma_grid):
        for j, lam in enumerate(lambda_grid):
            means, vars_, slopes = [], [], []
            for s in range(args.seeds):
                seed = 12345 + s
                series = run_closed_loop(config, gamma=float(gamma), lambda_=float(lam), seed=seed)
                m, v, sl = compute_stats(series, args.burn_in)
                means.append(m); vars_.append(v); slopes.append(sl)
            tau_mean[i, j] = float(np.mean(means))
            tau_var[i, j] = float(np.mean(vars_))
            tau_slope[i, j] = float(np.mean(slopes))

    np.savez_compressed(
        out_dir / "raw_runs" / "gamma_lambda_phase.npz",
        gamma_grid=gamma_grid,
        lambda_grid=lambda_grid,
        tau_mean=tau_mean,
        tau_var=tau_var,
        tau_slope=tau_slope,
        config=json.dumps(config, ensure_ascii=False)
    )
    print(f"[OK] Saved: {out_dir / 'raw_runs' / 'gamma_lambda_phase.npz'}")

if __name__ == "__main__":
    main()
