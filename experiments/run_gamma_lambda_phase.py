from __future__ import annotations
import time
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
    ap.add_argument("--gamma_points", type=int, default=2)
    ap.add_argument("--lambda_points", type=int, default=2)
    ap.add_argument("--lambda_max", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--burn_in", type=float, default=0.2)
    args = ap.parse_args()
    #
    # config = {
    #     "N": 15,
    #     "duration_min": 120.0,  # 真实 120 分钟
    #     "servers": 3,
    #     "mu": 1.2,  # users/min
    #     "base_rate": 3.0,  # NHPP base
    #     "peak_rate": 12.0,  # NHPP peak
    #     "sigma": 20.0,
    #     "K0": 3.0,
    #     "Kmax": 15.0,
    #     "eta_dist": (0.0, 10.0),  # travel ETA in minutes, uniform
    #     "mc_samples": 200,
    #     "burn_in": 0.2
    # }

    # 主程序中的 config 部分，修改为以下最小参数
    # config 调整为中等参数
    config = {
        "N": 8,  # 从2→8
        "duration_min": 30.0,  # 从5→30分钟
        "servers": 2,  # 从1→2
        "mu": 1.2,
        "base_rate": 2.0,  # 从1→2
        "peak_rate": 6.0,  # 从2→6
        "sigma": 20.0,
        "K0": 3.0,
        "Kmax": 15.0,
        "eta_dist": (0.0, 5.0),  # 从1→5分钟
        "mc_samples": 20,  # 从5→20
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

    # 新增：总任务数和进度计数
    total_tasks = len(gamma_grid) * len(lambda_grid) * args.seeds
    completed_tasks = 0
    start_time = time.time()

    for i, gamma in enumerate(gamma_grid):
        for j, lam in enumerate(lambda_grid):
            means, vars_, slopes = [], [], []
            for s in range(args.seeds):
                # 新增：打印当前进度
                completed_tasks += 1
                elapsed = time.time() - start_time
                remaining = (elapsed / completed_tasks) * (total_tasks - completed_tasks) if completed_tasks > 0 else 0
                print(f"[进度] γ={gamma:.2f}, λ={lam:.2f}, seed={s} | "
                      f"完成 {completed_tasks}/{total_tasks} ({completed_tasks / total_tasks * 100:.1f}%) | "
                      f"已耗时 {elapsed:.1f}s | 剩余约 {remaining:.1f}s", flush=True)

                seed = 12345 + s
                # 新增：单次模拟计时，定位慢函数
                sim_start = time.time()
                series = run_closed_loop(config, gamma=float(gamma), lambda_=float(lam), seed=seed)
                sim_elapsed = time.time() - sim_start
                print(f"  - 本次run_closed_loop耗时: {sim_elapsed:.1f}s", flush=True)

                m, v, sl = compute_stats(series, args.burn_in)
                means.append(m);
                vars_.append(v);
                slopes.append(sl)
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
