from __future__ import annotations
import time
import argparse
import json
from pathlib import Path
import numpy as np

from src.simulation.closed_loop import run_closed_loop


def compute_stats(tau_series: np.ndarray, burn_in_frac: float):
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
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--burn_in", type=float, default=0.2)
    args = ap.parse_args()
    #
    # config = {
    #     "N": 5,
    #     "duration_min": 60.0,
    #     "servers": 1,
    #     "mu": 1.0,
    #     "base_rate": 3.0,
    #     "peak_rate": 8.0,
    #     "sigma": 15.0,
    #     "K0": 3.0,
    #     "Kmax": 20.0,
    #     "eta_dist": (0.0, 8.0),
    #     "mc_samples": 20,
    #     "burn_in": 0.2
    # }
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
        "mc_samples": 200,
        "burn_in": 0.2
    }

    out_dir = Path(args.out_dir)
    (out_dir / "raw_runs").mkdir(parents=True, exist_ok=True)

    gamma_grid = np.linspace(0.0, 1.0, args.gamma_points)
    lambda_grid = np.linspace(0.0, args.lambda_max, args.lambda_points)

    tau_mean = np.zeros((len(gamma_grid), len(lambda_grid)))
    tau_var = np.zeros_like(tau_mean)
    tau_slope = np.zeros_like(tau_mean)

    total_tasks = len(gamma_grid) * len(lambda_grid) * args.seeds
    completed_tasks = 0
    start_time = time.time()

    for i, gamma in enumerate(gamma_grid):
        for j, lam in enumerate(lambda_grid):
            means, vars_, slopes = [], [], []

            for s in range(args.seeds):
                completed_tasks += 1

                elapsed = time.time() - start_time
                remaining = (elapsed / completed_tasks) * (total_tasks - completed_tasks)

                print(
                    f"[进度] γ={gamma:.2f}, λ={lam:.2f}, seed={s} | "
                    f"{completed_tasks}/{total_tasks} "
                    f"({completed_tasks / total_tasks * 100:.1f}%) | "
                    f"已耗时 {elapsed:.1f}s | 剩余约 {remaining:.1f}s",
                    flush=True
                )

                seed = 12345 + s
                sim_start = time.time()

                series, _ = run_closed_loop(
                    config,
                    gamma=float(gamma),
                    lambda_=float(lam),
                    seed=seed
                )

                sim_elapsed = time.time() - sim_start
                print(f"  - run_closed_loop耗时: {sim_elapsed:.2f}s", flush=True)

                m, v, sl = compute_stats(series, args.burn_in)
                means.append(m)
                vars_.append(v)
                slopes.append(sl)

            tau_mean[i, j] = np.mean(means)
            tau_var[i, j] = np.mean(vars_)
            tau_slope[i, j] = np.mean(slopes)

    np.savez_compressed(
        out_dir / "raw_runs" / "gamma_lambda_phase.npz",
        gamma_grid=gamma_grid,
        lambda_grid=lambda_grid,
        tau_mean=tau_mean,
        tau_var=tau_var,
        tau_slope=tau_slope,
        config=json.dumps(config, ensure_ascii=False)
    )

    print("[OK] Saved gamma_lambda_phase.npz")


if __name__ == "__main__":
    main()