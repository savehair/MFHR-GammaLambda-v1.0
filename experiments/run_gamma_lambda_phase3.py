from __future__ import annotations
import time
import argparse
import json
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count  # 新增：并行相关模块

from src.simulation.closed_loop import run_closed_loop


# 新增：封装单次仿真任务（必须定义在main外，支持多进程序列化）
def run_single_task(args_tuple):
    """
    单次仿真任务封装，供多进程调用
    args_tuple: (config, gamma, lambda_, seed, burn_in)
    """
    config, gamma, lambda_, seed, burn_in = args_tuple

    # 执行单次仿真
    series, _ = run_closed_loop(config, gamma=float(gamma), lambda_=float(lambda_), seed=seed)

    # 计算统计量
    burn = int(len(series) * burn_in)
    x = series[burn:]
    t = np.arange(len(x), dtype=float)
    slope, _ = np.polyfit(t, x, 1)
    mean = float(np.mean(x))
    var = float(np.var(x))
    slope = float(slope)

    return gamma, lambda_, seed, mean, var, slope


def compute_stats(tau_series: np.ndarray, burn_in_frac: float):
    burn = int(len(tau_series) * burn_in_frac)
    x = tau_series[burn:]
    t = np.arange(len(x), dtype=float)
    slope, _ = np.polyfit(t, x, 1)
    return float(np.mean(x)), float(np.var(x)), float(slope)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--gamma_points", type=int, default=21)
    ap.add_argument("--lambda_points", type=int, default=21)
    ap.add_argument("--lambda_max", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--burn_in", type=float, default=0.2)
    ap.add_argument("--n_workers", type=int, default=8, help="并行进程数，4")  # 新增
    args = ap.parse_args()

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
        "burn_in": 0.2
    }

    out_dir = Path(args.out_dir)
    (out_dir / "raw_runs").mkdir(parents=True, exist_ok=True)

    gamma_grid = np.linspace(0.0, 1.0, args.gamma_points)
    lambda_grid = np.linspace(0.0, args.lambda_max, args.lambda_points)

    # 步骤1：构建所有待执行的任务列表
    task_list = []
    for gamma in gamma_grid:
        for lam in lambda_grid:
            for s in range(args.seeds):
                seed = 12345 + s
                # 每个任务封装为元组（方便多进程传递）
                task = (config, gamma, lam, seed, args.burn_in)
                task_list.append(task)

    total_tasks = len(task_list)
    print(f"[并行模式] 共 {total_tasks} 个任务，使用 {args.n_workers} 个进程（M1共{cpu_count()}核）")

    # 步骤2：多进程执行所有任务
    start_time = time.time()
    results = []
    with Pool(processes=args.n_workers) as pool:  # 创建进程池
        # 逐个获取任务结果，同时打印进度
        for idx, result in enumerate(pool.imap_unordered(run_single_task, task_list), 1):
            gamma, lam, seed, mean, var, slope = result
            results.append(result)

            # 打印进度
            elapsed = time.time() - start_time
            remaining = (elapsed / idx) * (total_tasks - idx) if idx > 0 else 0
            print(
                f"[进度] γ={gamma:.2f}, λ={lam:.2f}, seed={seed - 12345} | "
                f"{idx}/{total_tasks} ({idx / total_tasks * 100:.1f}%) | "
                f"已耗时 {elapsed:.1f}s | 剩余约 {remaining:.1f}s",
                flush=True
            )

    # 步骤3：整理结果到数组（和原逻辑一致）
    tau_mean = np.zeros((len(gamma_grid), len(lambda_grid)))
    tau_var = np.zeros_like(tau_mean)
    tau_slope = np.zeros_like(tau_mean)

    # 构建索引映射：(gamma, lambda_) → (i,j)
    gamma_to_idx = {g: i for i, g in enumerate(gamma_grid)}
    lambda_to_idx = {l: j for j, l in enumerate(lambda_grid)}

    # 按种子分组统计
    temp_results = {}
    for gamma, lam, seed, mean, var, slope in results:
        key = (gamma, lam)
        if key not in temp_results:
            temp_results[key] = {"means": [], "vars": [], "slopes": []}
        temp_results[key]["means"].append(mean)
        temp_results[key]["vars"].append(var)
        temp_results[key]["slopes"].append(slope)

    # 计算多种子的均值
    for (gamma, lam), vals in temp_results.items():
        i = gamma_to_idx[gamma]
        j = lambda_to_idx[lam]
        tau_mean[i, j] = np.mean(vals["means"])
        tau_var[i, j] = np.mean(vals["vars"])
        tau_slope[i, j] = np.mean(vals["slopes"])

    # 保存结果（和原逻辑一致）
    np.savez_compressed(
        out_dir / "raw_runs" / "gamma_lambda_phase.npz",
        gamma_grid=gamma_grid,
        lambda_grid=lambda_grid,
        tau_mean=tau_mean,
        tau_var=tau_var,
        tau_slope=tau_slope,
        config=json.dumps(config, ensure_ascii=False)
    )

    total_elapsed = time.time() - start_time
    print(f"[OK] 所有任务完成！总耗时 {total_elapsed:.1f}s")
    print(f"[OK] Saved gamma_lambda_phase.npz")
    print(f"tau_mean min/max: {np.min(tau_mean)} {np.max(tau_mean)}")
    print(f"tau_var  min/max: {np.min(tau_var)} {np.max(tau_var)}")
    print(f"tau_slope min/max: {np.min(tau_slope)} {np.max(tau_slope)}")


if __name__ == "__main__":
    # macOS/M1 多进程必须加这行（避免递归创建进程）
    import multiprocessing

    multiprocessing.set_start_method('fork', force=True)
    main()