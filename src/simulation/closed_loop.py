# src/simulation/closed_loop.py
from __future__ import annotations
import numpy as np
from src.simulation.simpy_system import run_system

def run_closed_loop(
    config: dict,
    gamma: float,
    lambda_: float,
    seed: int,
    predictor=None,
    *,
    ABLATE_LAMBDA: bool = False,
    FIX_GAMMA: bool = False,
    NO_PRIORITY: bool = False,
    USE_DYNAMIC_LAMBDA: bool = False,
) -> tuple[np.ndarray, float]:
    """Run one closed-loop MFHR simulation.

    Returns:
        A tuple ``(tau_series, fairness)`` where ``tau_series`` is the per-minute
        average wait-time series and ``fairness`` is Jain's index over realized waits.
    """
    rng = np.random.default_rng(seed)

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
        predictor=predictor,
        ABLATE_LAMBDA=ABLATE_LAMBDA,
        FIX_GAMMA=FIX_GAMMA,
        NO_PRIORITY=NO_PRIORITY,
        USE_DYNAMIC_LAMBDA=USE_DYNAMIC_LAMBDA,
    )
