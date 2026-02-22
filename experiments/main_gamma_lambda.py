from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import numpy as np

from src.simulation.closed_loop import run_closed_loop


# ==============================
# τ 统计
# ==============================
def compute_stats(tau_series: np.ndarray, burn_in_frac: float):
    burn = int(len(tau_series) * burn_in_frac)
    x = tau_series[burn:]

    t = np.arange(len(x), dtype=float)
    slope, _ = np.polyfit(t, x, 1)

    return float(np.mean(x)), float(np.var(x)), float(slope)


# ==============================
# 单任务执行函数（并行核心）
# ==============================
def run_single_task(args_tuple):
    config, gamma, lam, seed, burn_in = args_tuple

    series = run_closed_loop(
        config=config,
        gamma=float(gamma),
        lambda_=float(lam),
        seed=seed
    )

    return compute_stats(series, burn_in)


# ==============================
# 主程序
# ==============================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--gamma_points", type=int, default=21)
    ap.add_argument("--lambda_points", type=int, default=21)
    ap.add_argument("--lambda_max", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--burn_in", type=float, default=0.2)
    ap.add_argument("--n_workers", type=int, default=4)

    args = ap.parse_args()

    # ==============================
    # config 真正进入系统
    # ==============================
    config = {
        "N": 15,
        "duration_min": 120.0,
        "servers": 3,
        "mu": 1.2,
        "base_rate": 3.0,
        "peak_rate": 12.0,
        "sigma": 20.0,
        "K0": 3.0,
        "Kmax": 15.0,
        "eta_dist": (0.0, 10.0),
        "mc_samples": 200,
        "burn_in": args.burn_in
    }

    out_dir = Path(args.out_dir)
    (out_dir / "raw_runs").mkdir(parents=True, exist_ok=True)

    gamma_grid = np.linspace(0.0, 1.0, args.gamma_points)
    lambda_grid = np.linspace(0.0, args.lambda_max, args.lambda_points)

    tau_mean = np.zeros((len(gamma_grid), len(lambda_grid)))
    tau_var = np.zeros_like(tau_mean)
    tau_slope = np.zeros_like(tau_mean)

    # ==============================
    # 构建任务列表
    # ==============================
    tasks = []

    for i, gamma in enumerate(gamma_grid):
        for j, lam in enumerate(lambda_grid):
            for s in range(args.seeds):
                seed = 12345 + s
                tasks.append((config, gamma, lam, seed, args.burn_in))

    print(f"总任务数: {len(tasks)}")
    print(f"使用进程数: {args.n_workers}")

    start_time = time.time()

    # ==============================
    # 并行执行
    # ==============================
    with Pool(processes=args.n_workers) as pool:
        results = pool.map(run_single_task, tasks)

    # ==============================
    # 聚合结果
    # ==============================
    idx = 0
    for i in range(len(gamma_grid)):
        for j in range(len(lambda_grid)):

            means = []
            vars_ = []
            slopes = []

            for _ in range(args.seeds):
                m, v, sl = results[idx]
                means.append(m)
                vars_.append(v)
                slopes.append(sl)
                idx += 1

            tau_mean[i, j] = float(np.mean(means))
            tau_var[i, j] = float(np.mean(vars_))
            tau_slope[i, j] = float(np.mean(slopes))

    # ==============================
    # 保存结果
    # ==============================
    np.savez_compressed(
        out_dir / "raw_runs" / "gamma_lambda_phase.npz",
        gamma_grid=gamma_grid,
        lambda_grid=lambda_grid,
        tau_mean=tau_mean,
        tau_var=tau_var,
        tau_slope=tau_slope,
        config=json.dumps(config)
    )

    elapsed = time.time() - start_time
    print(f"\n完成，总耗时: {elapsed:.2f}s")
    print("保存至:", out_dir / "raw_runs" / "gamma_lambda_phase.npz")

    print("tau_mean min/max:", tau_mean.min(), tau_mean.max())
    print("tau_var  min/max:", tau_var.min(), tau_var.max())
    print("tau_slope min/max:", tau_slope.min(), tau_slope.max())


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()