from __future__ import annotations
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.simulation.closed_loop import run_closed_loop
from src.metrics.fairness import jain_index


# ==============================
# τ 统计
# ==============================
def compute_stats(series, burn):
    burn_idx = int(len(series) * burn)
    x = series[burn_idx:]
    t = np.arange(len(x))
    slope, _ = np.polyfit(t, x, 1)
    return np.mean(x), np.var(x), slope


# ==============================
# 单任务执行
# ==============================
def run_task(args_tuple):
    try:
        config, gamma, lam, seed, burn, flags = args_tuple

        series, fairness = run_closed_loop(
            config=config,
            gamma=gamma,
            lambda_=lam,
            seed=seed,
            **flags
        )

        mean, var, slope = compute_stats(series, burn)
        return mean, var, slope, fairness

    except Exception as e:
        print("Task failed:", e)
        return 0.0, 0.0, 0.0, 1.0
    mean, var, slope = compute_stats(series, burn)
    return mean, var, slope, fairness


# ==============================
# 相图绘制
# ==============================
def plot_heatmap(matrix, title, path):
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, origin="lower", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ==============================
# 四代表点 τ(t)
# ==============================
def plot_four_cases(config, out_dir):

    cases = {
        "Stable": (0.0, 0.0),
        "Drift": (1.0, 0.0),
        "HighRisk": (1.0, 2.0),
        "Oscillatory": (0.5, 1.5),
    }

    plt.figure(figsize=(7, 5))

    for name, (g, l) in cases.items():
        series, _ = run_closed_loop(
            config=config,
            gamma=g,
            lambda_=l,
            seed=123,
        )
        plt.plot(series, label=name)

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("τ")
    plt.tight_layout()
    plt.savefig(out_dir / "four_cases_tau.png")
    plt.close()


# ==============================
# 主程序
# ==============================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--gamma_points", type=int, default=21)
    parser.add_argument("--lambda_points", type=int, default=21)
    parser.add_argument("--lambda_max", type=float, default=2.0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--burn_in", type=float, default=0.2)
    parser.add_argument("--n_workers", type=int, default=4)
    args = parser.parse_args()

    # ==============================
    # 系统参数
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
    out_dir.mkdir(exist_ok=True)

    gamma_grid = np.linspace(0, 1, args.gamma_points)
    lambda_grid = np.linspace(0, args.lambda_max, args.lambda_points)

    tau_mean = np.zeros((len(gamma_grid), len(lambda_grid)))
    tau_var = np.zeros_like(tau_mean)
    tau_slope = np.zeros_like(tau_mean)
    fairness_map = np.zeros_like(tau_mean)



    flags = {
        "ABLATE_LAMBDA": False,
        "FIX_GAMMA": False,
        "NO_PRIORITY": False,
        "USE_DYNAMIC_LAMBDA": False
    }

    tasks = []

    for i, g in enumerate(gamma_grid):
        for j, l in enumerate(lambda_grid):
            for s in range(args.seeds):
                seed = 12345 + s
                tasks.append((config, g, l, seed, args.burn_in, flags))

    with mp.Pool(args.n_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(run_task, tasks),
                total=len(tasks),
                desc="Gamma-Lambda Scan",
                ncols=100
            )
        )

    idx = 0
    for i in range(len(gamma_grid)):
        for j in range(len(lambda_grid)):

            means, vars_, slopes, fairs = [], [], [], []

            for _ in range(args.seeds):
                m, v, sl, f = results[idx]
                means.append(m)
                vars_.append(v)
                slopes.append(sl)
                fairs.append(f)
                idx += 1

            tau_mean[i, j] = np.mean(means)
            tau_var[i, j] = np.mean(vars_)
            tau_slope[i, j] = np.mean(slopes)
            fairness_map[i, j] = np.mean(fairs)

    # ==============================
    # 保存数据
    # ==============================
    np.savez_compressed(
        out_dir / "phase_results.npz",
        gamma_grid=gamma_grid,
        lambda_grid=lambda_grid,
        tau_mean=tau_mean,
        tau_var=tau_var,
        tau_slope=tau_slope,
        fairness=fairness_map,
        config=json.dumps(config)
    )

    # ==============================
    # 生成图像
    # ==============================
    plot_heatmap(tau_mean, "Tau Mean", out_dir / "heat_tau_mean.png")
    plot_heatmap(tau_var, "Tau Var", out_dir / "heat_tau_var.png")
    plot_heatmap(tau_slope, "Tau Slope", out_dir / "heat_tau_slope.png")
    plot_heatmap(fairness_map, "Jain Fairness", out_dir / "heat_fairness.png")

    plot_four_cases(config, out_dir)

    print("Phase 4 完成")
    print("tau_mean min/max:", tau_mean.min(), tau_mean.max())
    print("fairness min/max:", fairness_map.min(), fairness_map.max())


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()