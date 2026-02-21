# src/prediction/quantile_mc.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class JobSnapshot:
    # remaining service time (minutes)
    rem: float
    # priority key = arrival_time - K_used_at_arrival (smaller = earlier service)
    prio: float

def predict_wait_quantiles_mc(
    now: float,
    eta: float,
    in_service: List[JobSnapshot],
    queue: List[JobSnapshot],
    servers: int,
    mu: float,
    K_assumed: float,
    n_mc: int = 200,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float]:
    """
    Predict waiting at arrival time (now+eta) for a hypothetical user.
    Monte Carlo "shadow" simulation using current snapshot (no future arrivals included).

    Scheduling: smallest prio first.
    New job priority assumed as (arrival_time - K_assumed).

    Returns: (P50, P90) waiting in minutes.
    """
    if rng is None:
        rng = np.random.default_rng()

    arrival_t = now + eta
    waits = np.zeros(n_mc, dtype=float)

    for k in range(n_mc):
        # Copy snapshots
        ins = [JobSnapshot(rem=j.rem, prio=j.prio) for j in in_service]
        q = [JobSnapshot(rem=j.rem, prio=j.prio) for j in queue]

        # Simulate service completions during travel eta
        # Servers run in parallel: decrement remaining times by eta, pop completed and refill from queue by priority
        t_left = eta
        while t_left > 1e-9:
            if len(ins) == 0:
                break
            # next completion among in-service
            dt = min(j.rem for j in ins)
            if dt >= t_left:
                for j in ins:
                    j.rem -= t_left
                t_left = 0.0
                break
            # advance to completion
            for j in ins:
                j.rem -= dt
            t_left -= dt
            # remove completed jobs
            ins = [j for j in ins if j.rem > 1e-9]
            # refill idle servers from queue (priority)
            while len(ins) < servers and len(q) > 0:
                q.sort(key=lambda x: x.prio)
                nxt = q.pop(0)
                ins.append(nxt)

        # Insert hypothetical arrival at time arrival_t
        new_job = JobSnapshot(rem=float(rng.exponential(1.0 / mu)), prio=float(arrival_t - K_assumed))
        q.append(new_job)

        # Compute waiting time until new_job starts service:
        # continue serving with priority discipline, no new arrivals
        t = 0.0
        # If there is an idle server now, new_job may start immediately after priority comparison
        while True:
            # refill idle servers
            while len(ins) < servers and len(q) > 0:
                q.sort(key=lambda x: x.prio)
                nxt = q.pop(0)
                ins.append(nxt)

            # if new_job is in service -> waiting ends
            if any(obj is new_job for obj in ins):
                waits[k] = t
                break

            # advance to next completion
            dt = min(j.rem for j in ins) if len(ins) > 0 else 0.0
            if dt <= 0.0:
                # no in service but new_job not started implies no servers? shouldn't happen
                waits[k] = t
                break
            for j in ins:
                j.rem -= dt
            t += dt
            ins = [j for j in ins if j.rem > 1e-9]

    return float(np.quantile(waits, 0.50)), float(np.quantile(waits, 0.90))