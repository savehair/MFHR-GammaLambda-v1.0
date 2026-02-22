# src/simulation/closed_loop.py
from __future__ import annotations
import numpy as np
from src.simulation.simpy_system import run_system
import time

def run_closed_loop(config: dict, gamma: float, lambda_: float, seed: int,predictor=None) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # 新增：打印进入run_system前的日志
    print(f"  [closed_loop] 开始调用run_system | γ={gamma:.2f}, λ={lambda_:.2f}, seed={seed}", flush=True)
    start = time.time()

    return run_system(
        N=int(config["N"]),
        T=float(config["duration_min"]),
        servers=int(config["servers"]),
        mu=float(config["mu"]),
        base_rate=float(config["base_rate"]),
        peak_rate=float(config["peak_rate"]),
        sigma=float(config["sigma"]),
        gamma=float(gamma),
        lam=float(lambda_),
        K0=float(config["K0"]),
        Kmax=float(config["Kmax"]),
        eta_dist=tuple(config["eta_dist"]),
        seeds=1,
        burn_in=float(config.get("burn_in", 0.2)),
        mc_samples=int(config.get("mc_samples", 200)),
        rng=rng,
    )