from __future__ import annotations
import heapq
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class JobSnapshot:
    # 正在服务中的剩余服务时间（分钟）；排队快照可先记为 0，入服时再采样
    rem: float
    # 优先级键，值越小越早被服务
    prio: float


def _sample_service_time(mu: float, rng: np.random.Generator) -> float:
    # 指数分布：rate=mu（人/分钟），均值为 1/mu 分钟
    return float(rng.exponential(1.0 / mu))


def _build_queue_heap(queue: List[JobSnapshot]) -> list[tuple[float, int, JobSnapshot]]:
    """将排队列表构建为最小堆，避免在高频循环中反复 sort。"""
    heap = [(float(job.prio), idx, job) for idx, job in enumerate(queue)]
    heapq.heapify(heap)
    return heap


def _pop_next_job(queue_heap: list[tuple[float, int, JobSnapshot]]) -> JobSnapshot:
    return heapq.heappop(queue_heap)[2]


def _push_job(
    queue_heap: list[tuple[float, int, JobSnapshot]],
    job: JobSnapshot,
    seq: int,
) -> None:
    heapq.heappush(queue_heap, (float(job.prio), seq, job))


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
    预测一个“假设顾客”在到店时刻（now+eta）的等待时间分位数。
    方法：基于当前快照执行 Monte Carlo 影子仿真。

    关键点：
    - 队列中的顾客在被拉入服务时必须重新采样服务时长；
      否则 rem=0 会导致“瞬时完成”，破坏排队效应。

    调度规则：prio 越小越先服务。
    新顾客优先级：arrival_time - K_assumed。

    返回：(P50, P90)，单位分钟。
    """
    if rng is None:
        rng = np.random.default_rng()

    arrival_t = float(now + eta)
    waits = np.zeros(n_mc, dtype=float)

    for k in range(n_mc):
        # 深拷贝快照，避免污染真实系统状态
        ins = [JobSnapshot(rem=float(j.rem), prio=float(j.prio)) for j in in_service]
        q = [JobSnapshot(rem=float(j.rem), prio=float(j.prio)) for j in queue]
        queue_heap = _build_queue_heap(q)
        next_seq = len(queue_heap)

        # --- 1) 旅行期演化：从 now 推进到 arrival_t ---
        t_left = float(eta)

        # 过滤掉数值误差导致的非正剩余时长
        ins = [j for j in ins if j.rem > 1e-12]

        while t_left > 1e-9:
            # 先补满空闲服务台，再推进时间
            while len(ins) < servers and len(queue_heap) > 0:
                nxt = _pop_next_job(queue_heap)
                # 队列快照通常不带 rem，入服时采样服务时长
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

            # 推进到下一次服务完成事件
            for j in ins:
                j.rem -= dt
            t_left -= dt

            # 移除完成服务的顾客
            ins = [j for j in ins if j.rem > 1e-9]

        # --- 2) 在 arrival_t 插入假设顾客 ---
        new_job = JobSnapshot(
            rem=_sample_service_time(mu, rng),
            prio=float(arrival_t - K_assumed),
        )
        _push_job(queue_heap, new_job, next_seq)
        next_seq += 1

        # --- 3) 推进系统直到新顾客开始服务，累计等待时间 ---
        t = 0.0
        while True:
            while len(ins) < servers and len(queue_heap) > 0:
                nxt = _pop_next_job(queue_heap)
                # 从队列拉入服务时，若 rem 缺失则补采样
                if nxt.rem <= 1e-12:
                    nxt.rem = _sample_service_time(mu, rng)
                ins.append(nxt)

            # 新顾客已经入服，等待结束
            if any(obj is new_job for obj in ins):
                waits[k] = t
                break

            if len(ins) == 0:
                waits[k] = t
                break

            dt = min(j.rem for j in ins)
            if dt <= 1e-12:
                # 数值保护：出现异常小步长时直接退出，避免死循环
                waits[k] = t
                break

            for j in ins:
                j.rem -= dt
            t += dt
            ins = [j for j in ins if j.rem > 1e-9]

    return float(np.quantile(waits, 0.50)), float(np.quantile(waits, 0.90))
