from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class JobSnapshot:
    # remaining service time (minutes) if in_service; for queued jobs this can be 0 at snapshot
    rem: float
    # priority key (smaller served earlier)
    prio: float


def _sample_service_time(mu: float, rng: np.random.Generator) -> float:
    # exponential with rate mu (users/min) => mean 1/mu minutes
    return float(rng.exponential(1.0 / mu))


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
    Predict waiting time at arrival time (now+eta) for a hypothetical user.
    Monte Carlo shadow simulation using current snapshot.

    IMPORTANT FIX:
    - When a queued job is pulled into service in the shadow sim, we MUST sample its service time.
      Otherwise queued jobs have rem=0 and become instantaneous, destroying queue effects.

    Scheduling: smallest prio first.
    New job priority assumed as (arrival_time - K_assumed).

    Returns: (P50, P90) waiting minutes.
    """
    if rng is None:
        rng = np.random.default_rng()

    arrival_t = float(now + eta)
    waits = np.zeros(n_mc, dtype=float)

    for k in range(n_mc):
        # Deep copy snapshots
        ins = [JobSnapshot(rem=float(j.rem), prio=float(j.prio)) for j in in_service]
        q = [JobSnapshot(rem=float(j.rem), prio=float(j.prio)) for j in queue]

        # --- 1) Evolve system during travel time eta ---
        t_left = float(eta)

        # Ensure in-service jobs have positive rem
        ins = [j for j in ins if j.rem > 1e-12]

        while t_left > 1e-9:
            # Fill idle servers from queue BEFORE progressing time
            while len(ins) < servers and len(q) > 0:
                q.sort(key=lambda x: x.prio)
                nxt = q.pop(0)
                # queued jobs don't have remaining time -> sample fresh service time
                nxt.rem = _sample_service_time(mu, rng)
                ins.append(nxt)

            if len(ins) == 0:
                break

            dt = min(j.rem for j in ins)
            if dt >= t_left:
                for j in ins:
                    j.rem -= t_left
                t_left = 0.0
                break

            # advance to next completion
            for j in ins:
                j.rem -= dt
            t_left -= dt

            # remove completed jobs
            ins = [j for j in ins if j.rem > 1e-9]

        # --- 2) Insert hypothetical arrival at time arrival_t ---
        new_job = JobSnapshot(
            rem=_sample_service_time(mu, rng),
            prio=float(arrival_t - K_assumed),
        )
        q.append(new_job)

        # --- 3) Compute waiting time until new_job starts service ---
        t = 0.0
        while True:
            # Fill idle servers
            while len(ins) < servers and len(q) > 0:
                q.sort(key=lambda x: x.prio)
                nxt = q.pop(0)
                # If job came from queue, ensure rem sampled
                if nxt.rem <= 1e-12:
                    nxt.rem = _sample_service_time(mu, rng)
                ins.append(nxt)

            # If new_job in service -> waiting ends
            if any(obj is new_job for obj in ins):
                waits[k] = t
                break

            if len(ins) == 0:
                waits[k] = t
                break

            dt = min(j.rem for j in ins)
            if dt <= 1e-12:
                # numerical guard: if something went wrong, stop
                waits[k] = t
                break

            for j in ins:
                j.rem -= dt
            t += dt
            ins = [j for j in ins if j.rem > 1e-9]

    return float(np.quantile(waits, 0.50)), float(np.quantile(waits, 0.90))