import simpy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

# ===== 预测模块 =====
from src.prediction.quantile_mc import predict_wait_quantiles_mc
from src.selection.risk_choice import choose_model
from src.mfhr.k_controller import compute_K
from src.metrics.fairness import jain_index
from src.control.congestion_correction import dynamic_lambda
from src.prediction.quantile_mc import JobSnapshot

@dataclass
class StoreState:
    res: simpy.PriorityResource
    in_service: List = field(default_factory=list)
    queue: List = field(default_factory=list)

def nhpp_rate(t: float, base: float, peak: float, sigma: float, T: float) -> float:
    return base + peak * np.exp(-((t - T / 2.0) ** 2) / (2.0 * sigma ** 2))


def nhpp_generate_arrivals(env, T, base, peak, sigma, rng):
    lam_max = base + peak
    t = 0.0
    while t < T:
        t += -np.log(rng.random()) / lam_max
        if t >= T:
            break
        if rng.random() < nhpp_rate(t, base, peak, sigma, T) / lam_max:
            yield t

def service_proc(env, store, mu, job, req, rng):
    store.queue = [x for x in store.queue if x is not job]
    store.in_service.append(job)

    st = float(rng.exponential(1.0 / mu))
    job.rem = st
    yield env.timeout(st)

    store.in_service = [x for x in store.in_service if x is not job]
    store.res.release(req)

#接入sti
def build_features(now, eta, store):
    """
    简化特征：
    [当前时间, ETA, 队列长度, 服务中人数]
    """
    return np.array([
        now,
        eta,
        len(store.queue),
        len(store.in_service)
    ], dtype=np.float32)
def run_system(
        *,
        N: int,
        T: float,
        servers: int,
        mu: float,
        base_rate: float,
        peak_rate: float,
        sigma: float,
        gamma: float,
        lam: float,
        K0: float,
        Kmax: float,
        eta_dist: Tuple[float, float],
        seeds: int,
        burn_in: float,
        mc_samples: int,
        rng: np.random.Generator,

        # ===== 新增 =====
        predictor=None,
        ABLATE_LAMBDA=False,
        FIX_GAMMA=False,
        NO_PRIORITY=False,
        USE_DYNAMIC_LAMBDA=False,
):
    env = simpy.Environment()
    stores = [StoreState(res=simpy.PriorityResource(env, capacity=servers))
              for _ in range(N)]

    bucket = int(np.ceil(T))
    waits_by_minute = [[] for _ in range(bucket + 1)]
    all_waits = []

    def customer_decision_process(arrival_t: float, cust_id: int):

        eta = float(rng.uniform(eta_dist[0], eta_dist[1]))
        now = arrival_t

        W50 = np.zeros(N)
        W90 = np.zeros(N)

        for j in range(N):

            st = stores[j]

            # =============================
            # 第一轮预测（基准K0）
            # =============================
            if predictor is not None:
                features = build_features(now, eta, st)  # 你已有特征构造
                p50_0, p90_0 = predictor.predict_quantiles(features)
            else:
                p50_0, p90_0 = predict_wait_quantiles_mc(
                    now=now,
                    eta=eta,
                    in_service=st.in_service,
                    queue=st.queue,
                    servers=servers,
                    mu=mu,
                    K_assumed=K0,
                    n_mc=mc_samples,
                    rng=rng,
                )

            delta = p90_0 - p50_0

            # =============================
            # λ 消融
            # =============================
            if ABLATE_LAMBDA:
                K_j = K0
            else:
                if USE_DYNAMIC_LAMBDA:
                    lam_eff = dynamic_lambda(lam, p90_0, threshold=10)
                else:
                    lam_eff = lam

                K_j = compute_K(K0, lam_eff, delta, Kmax)

            # =============================
            # 第二轮预测（按 K_j）
            # =============================
            if predictor is not None:
                # 当前 predictor 接口仅依赖 features，不依赖 K_assumed；
                # 直接复用第一轮结果，避免重复推理开销。
                p50, p90 = p50_0, p90_0
            elif lam == 0 or ABLATE_LAMBDA:
                # λ=0 或 λ 消融时，K_j 与 K0 等价，不需要重复 MC 预测。
                p50, p90 = p50_0, p90_0
            else:
                p50, p90 = predict_wait_quantiles_mc(
                    now=now,
                    eta=eta,
                    in_service=st.in_service,
                    queue=st.queue,
                    servers=servers,
                    mu=mu,
                    K_assumed=K_j,
                    n_mc=mc_samples,
                    rng=rng,
                )

            W50[j] = p50
            W90[j] = p90

        # =============================
        # γ 消融
        # =============================
        gamma_eff = 0.0 if FIX_GAMMA else gamma
        j_star, delta_all = choose_model(W50, W90, gamma_eff)

        delta_j = float(delta_all[j_star])

        if ABLATE_LAMBDA:
            K_j = K0
        else:
            K_j = compute_K(K0, lam, delta_j, Kmax)

        store_arrival_t = now + eta
        prio = float(store_arrival_t - K_j)

        yield env.timeout(eta)

        # =============================
        # Priority 消融
        # =============================
        if NO_PRIORITY:
            req = stores[j_star].res.request()
        else:
            req = stores[j_star].res.request(priority=prio)

        job = JobSnapshot(rem=0.0, prio=prio)
        stores[j_star].queue.append(job)

        yield req

        start_t = env.now
        w_real = float(start_t - store_arrival_t)

        all_waits.append(w_real)

        m = int(np.floor(start_t))
        if 0 <= m <= bucket:
            waits_by_minute[m].append(w_real)

        env.process(service_proc(env, stores[j_star], mu, job, req, rng))

    # =============================
    # 生成到达
    # =============================
    arr_times = list(nhpp_generate_arrivals(env, T, base_rate,
                                            peak_rate, sigma, rng))

    for cid, t in enumerate(arr_times):
        def launch(tt=t, cc=cid):
            yield env.timeout(tt)
            yield env.process(customer_decision_process(tt, cc))
        env.process(launch())

    env.run(until=T)

    # =============================
    # τ 统计
    # =============================
    tau_series = np.full(bucket, np.nan)

    for m in range(bucket):
        if waits_by_minute[m]:
            tau_series[m] = float(np.mean(waits_by_minute[m]))

    # forward fill
    last = None
    for i in range(len(tau_series)):
        if np.isnan(tau_series[i]):
            if last is not None:
                tau_series[i] = last
        else:
            last = tau_series[i]

    # back fill
    first = None
    for i in range(len(tau_series)):
        if not np.isnan(tau_series[i]):
            first = tau_series[i]
            break

    if first is not None:
        for i in range(len(tau_series)):
            if np.isnan(tau_series[i]):
                tau_series[i] = first
            else:
                break

    tau_series = np.nan_to_num(tau_series, nan=0.0)

    fairness = jain_index(all_waits) if all_waits else 1.0

    return tau_series, fairness
# @dataclass
# class StoreState:
#     res: simpy.PriorityResource
#     in_service: List[JobSnapshot] = field(default_factory=list)
#     queue: List[JobSnapshot] = field(default_factory=list)
#
#
# def nhpp_rate(t: float, base: float, peak: float, sigma: float, T: float) -> float:
#     return base + peak * np.exp(-((t - T / 2.0) ** 2) / (2.0 * sigma ** 2))
#
#
# def nhpp_generate_arrivals(env, T, base, peak, sigma, rng):
#     lam_max = base + peak
#     t = 0.0
#     while t < T:
#         t += -np.log(rng.random()) / lam_max
#         if t >= T:
#             break
#         if rng.random() < nhpp_rate(t, base, peak, sigma, T) / lam_max:
#             yield t
#
#
# def service_proc(env, store, mu, job, req, rng):
#     store.queue = [x for x in store.queue if x is not job]
#     store.in_service.append(job)
#
#     st = float(rng.exponential(1.0 / mu))
#     job.rem = st
#     yield env.timeout(st)
#
#     store.in_service = [x for x in store.in_service if x is not job]
#     store.res.release(req)
#
#
# def run_system(
#     *,
#     N: int,
#     T: float,
#     servers: int,
#     mu: float,
#     base_rate: float,
#     peak_rate: float,
#     sigma: float,
#     gamma: float,
#     lam: float,
#     K0: float,
#     Kmax: float,
#     eta_dist: Tuple[float, float],
#     seeds: int,
#     burn_in: float,
#     mc_samples: int,
#     rng: np.random.Generator,
#
#     predictor=None,
#     ABLATE_LAMBDA=False,
#     FIX_GAMMA=False,
#     NO_PRIORITY=False,
#     USE_DYNAMIC_LAMBDA=False,
# ) -> np.ndarray:
#
#     env = simpy.Environment()
#     stores = [StoreState(res=simpy.PriorityResource(env, capacity=servers)) for _ in range(N)]
#
#     bucket = int(np.ceil(T))
#     waits_by_minute = [[] for _ in range(bucket + 1)]
#
#     def customer_decision_process(arrival_t: float, cust_id: int):
#         eta = float(rng.uniform(eta_dist[0], eta_dist[1]))
#         now = arrival_t
#
#         W50 = np.zeros(N)
#         W90 = np.zeros(N)
#
#         # ====== 严格闭环预测 ======
#         for j in range(N):
#             st = stores[j]
#
#             # --- 第一轮预测：用 K0 ---
#             p50_0, p90_0 = predict_wait_quantiles_mc(
#                 now=now,
#                 eta=eta,
#                 in_service=st.in_service,
#                 queue=st.queue,
#                 servers=servers,
#                 mu=mu,
#                 K_assumed=K0,
#                 n_mc=mc_samples,
#                 rng=rng,
#             )
#
#             delta = p90_0 - p50_0
#             K_j = compute_K(K0, lam, delta, Kmax)
#
#             # --- 第二轮预测：用 K_j ---
#             p50, p90 = predict_wait_quantiles_mc(
#                 now=now,
#                 eta=eta,
#                 in_service=st.in_service,
#                 queue=st.queue,
#                 servers=servers,
#                 mu=mu,
#                 K_assumed=K_j,
#                 n_mc=mc_samples,
#                 rng=rng,
#             )
#
#             W50[j] = p50
#             W90[j] = p90
#
#         # ====== 用户选择 ======
#         j_star, delta_all = choose_model(W50, W90, gamma)
#
#         delta_j = float(delta_all[j_star])
#         K_j = compute_K(K0, lam, delta_j, Kmax)
#
#         store_arrival_t = now + eta
#         prio = float(store_arrival_t - K_j)
#
#         yield env.timeout(eta)
#
#         req = stores[j_star].res.request(priority=prio)
#         job = JobSnapshot(rem=0.0, prio=prio)
#         stores[j_star].queue.append(job)
#
#         yield req
#
#         start_t = env.now
#         w_real = float(start_t - store_arrival_t)
#
#         m = int(np.floor(start_t))
#         if 0 <= m <= bucket:
#             waits_by_minute[m].append(w_real)
#
#         env.process(service_proc(env, stores[j_star], mu, job, req, rng))
#
#     # ====== 生成到达 ======
#     arr_times = list(nhpp_generate_arrivals(env, T, base_rate, peak_rate, sigma, rng))
#
#     for cid, t in enumerate(arr_times):
#         def launch(tt=t, cc=cid):
#             yield env.timeout(tt)
#             yield env.process(customer_decision_process(tt, cc))
#         env.process(launch())
#
#     env.run(until=T)
#
#     # ====== 计算 τ 时间序列 ======
#     tau_series = np.full(bucket, np.nan)
#
#     for m in range(bucket):
#         if waits_by_minute[m]:
#             tau_series[m] = float(np.mean(waits_by_minute[m]))
#
#     # forward fill
#     last = None
#     for i in range(len(tau_series)):
#         if np.isnan(tau_series[i]):
#             if last is not None:
#                 tau_series[i] = last
#         else:
#             last = tau_series[i]
#
#     # back fill
#     first = None
#     for i in range(len(tau_series)):
#         if not np.isnan(tau_series[i]):
#             first = tau_series[i]
#             break
#
#     if first is not None:
#         for i in range(len(tau_series)):
#             if np.isnan(tau_series[i]):
#                 tau_series[i] = first
#             else:
#                 break
#
#     tau_series = np.nan_to_num(tau_series, nan=0.0)
#
#     return tau_series
# # src/simulation/simpy_system.py
# from __future__ import annotations
# import simpy
# import numpy as np
# from dataclasses import dataclass, field
# from typing import List, Tuple, Dict
#
# from src.prediction.quantile_mc import JobSnapshot, predict_wait_quantiles_mc
# from src.selection.risk_choice import choose_model
# from src.mfhr.k_controller import compute_K
#
# @dataclass
# class StoreState:
#     res: simpy.PriorityResource
#     # snapshots (for prediction)
#     in_service: List[JobSnapshot] = field(default_factory=list)
#     queue: List[JobSnapshot] = field(default_factory=list)
#
# def nhpp_rate(t: float, base: float, peak: float, sigma: float, T: float) -> float:
#     return base + peak * np.exp(-((t - T / 2.0) ** 2) / (2.0 * sigma ** 2))
#
# def nhpp_generate_arrivals(env: simpy.Environment, T: float, base: float, peak: float, sigma: float, rng: np.random.Generator):
#     """Thinning algorithm to yield arrival times in [0, T]."""
#     lam_max = base + peak
#     t = 0.0
#     while t < T:
#         u = rng.random()
#         t += -np.log(u) / lam_max
#         if t >= T:
#             break
#         if rng.random() < nhpp_rate(t, base, peak, sigma, T) / lam_max:
#             yield t
#
# def service_proc(env: simpy.Environment, store: StoreState, mu: float, job: JobSnapshot, req: simpy.events.Event, rng: np.random.Generator):
#     """Run one job after resource acquired."""
#     # Move job into in_service snapshot (remove from queue snapshot if present)
#     store.queue = [x for x in store.queue if x is not job]
#     store.in_service.append(job)
#
#     # service time: exponential with rate mu
#     st = float(rng.exponential(1.0 / mu))
#     job.rem = st
#     yield env.timeout(st)
#
#     # remove from in_service
#     store.in_service = [x for x in store.in_service if x is not job]
#     store.res.release(req)
#
# def arrive_to_store(env: simpy.Environment, store: StoreState, mu: float, prio: float, rng: np.random.Generator):
#     """Customer arrives physically to chosen store and requests service."""
#     job = JobSnapshot(rem=0.0, prio=prio)
#     store.queue.append(job)
#     req = store.res.request(priority=prio)
#     yield req
#     # start service
#     env.process(service_proc(env, store, mu, job, req, rng))
#
# def run_system(
#     *,
#     N: int,
#     T: float,
#     servers: int,
#     mu: float,
#     base_rate: float,
#     peak_rate: float,
#     sigma: float,
#     gamma: float,
#     lam: float,
#     K0: float,
#     Kmax: float,
#     eta_dist: Tuple[float, float],  # uniform ETA range (min,max)
#     seeds: int,
#     burn_in: float,
#     mc_samples: int,
#     rng: np.random.Generator,
# ) -> np.ndarray:
#     """
#     Returns tau time series sampled every 1 minute (can adjust).
#     tau = mean realized waiting time of customers who START service within that minute window.
#     """
#
#     env = simpy.Environment()
#     stores = [StoreState(res=simpy.PriorityResource(env, capacity=servers)) for _ in range(N)]
#
#     # record realized waits by service-start minute bucket
#     bucket = int(np.ceil(T))
#     waits_by_minute: List[List[float]] = [[] for _ in range(bucket + 1)]
#
#     def customer_decision_process(arrival_t: float, cust_id: int):
#         # Customer appears at decision time arrival_t (NOT at store)
#         # Draw ETA (travel time)
#         eta = float(rng.uniform(eta_dist[0], eta_dist[1]))
#         now = arrival_t
#
#         # Step 1: predict (P50,P90) for each store as arrival-time wait at (now+eta)
#         W50 = np.zeros(N, dtype=float)
#         W90 = np.zeros(N, dtype=float)
#
#         for j in range(N):
#             st = stores[j]
#             p50, p90 = predict_wait_quantiles_mc(
#                 now=now,
#                 eta=eta,
#                 in_service=st.in_service,
#                 queue=st.queue,
#                 servers=servers,
#                 mu=mu,
#                 K_assumed=K0,          # for choice stage assume baseline K0
#                 n_mc=mc_samples,
#                 rng=rng
#             )
#             W50[j], W90[j] = p50, p90
#
#         # Step 2: user choice
#         j_star, delta_all = choose_model(W50, W90, gamma)
#
#         # Step 3: map uncertainty to K, then set priority shift
#         delta_j = float(delta_all[j_star])
#         K_j = compute_K(K0, lam, delta_j, Kmax)
#
#         # Priority discipline: priority = arrival_time_at_store - K_j
#         store_arrival_t = now + eta
#         prio = float(store_arrival_t - K_j)
#
#         # schedule physical arrival at store
#         yield env.timeout(eta)
#
#         # record waiting time: when service starts, we can compute start_time - store_arrival_t
#         req = stores[j_star].res.request(priority=prio)
#         job = JobSnapshot(rem=0.0, prio=prio)
#         stores[j_star].queue.append(job)
#
#         yield req
#         start_t = env.now
#         w_real = float(start_t - store_arrival_t)
#         m = int(np.floor(start_t))
#         if 0 <= m <= bucket:
#             waits_by_minute[m].append(w_real)
#
#         # start service
#         env.process(service_proc(env, stores[j_star], mu, job, req, rng))
#
#     # generate arrival times and start decision processes
#     arr_times = list(nhpp_generate_arrivals(env, T, base_rate, peak_rate, sigma, rng))
#     for cid, t in enumerate(arr_times):
#         def launch(tt=t, cc=cid):
#             yield env.timeout(tt)
#             yield env.process(customer_decision_process(tt, cc))
#         env.process(launch())
#
#     # run
#     env.run(until=T)
#
#     # compute tau series per minute
#     tau_series = np.full(bucket, np.nan, dtype=float)
#     for m in range(bucket):
#         if len(waits_by_minute[m]) > 0:
#             tau_series[m] = float(np.mean(waits_by_minute[m]))
#
#     # fill missing minutes by forward-fill then back-fill (to fit slope stats)
#     # (for plotting & phase metrics; does not affect mean much)
#     if np.all(np.isnan(tau_series)):
#         tau_series[:] = 0.0
#     else:
#         # forward fill
#         last = None
#         for i in range(len(tau_series)):
#             if np.isnan(tau_series[i]):
#                 if last is not None:
#                     tau_series[i] = last
#             else:
#                 last = tau_series[i]
#         # back fill
#         first = None
#         for i in range(len(tau_series)):
#             if not np.isnan(tau_series[i]):
#                 first = tau_series[i]
#                 break
#         if first is not None:
#             for i in range(len(tau_series)):
#                 if np.isnan(tau_series[i]):
#                     tau_series[i] = first
#                 else:
#                     break
#
#     return tau_series
