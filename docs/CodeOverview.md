# 代码阅读与结构理解

本文档总结仓库核心模块、执行链路和关键数据流，便于快速上手二次开发与实验复现。

## 1. 项目目标与总体流程

该项目围绕 **MFHR（风险感知排队调度）闭环系统** 展开，核心目标是：

1. 在非平稳到达流（NHPP）下模拟多门店/多服务台排队系统；
2. 通过预测模块估计等待时间分位数（`W50/W90`）；
3. 由选择策略 `gamma`（风险偏好）和调度参数 `lambda`（拥塞修正强度）共同决定用户分配与优先级；
4. 扫描 `gamma-lambda` 相图并统计系统稳定性与公平性指标。

主实验入口是 `experiments/main_experiment.py`：并行扫描参数网格，调用 `run_closed_loop`，再输出热图和四类代表轨迹图。

---

## 2. 关键目录说明

- `src/simulation/`
  - `closed_loop.py`：闭环仿真封装，负责把配置映射到系统模拟器；
  - `simpy_system.py`：核心离散事件仿真（SimPy），实现到达生成、排队、服务、调度策略。
- `src/prediction/`
  - `quantile_mc.py`：MC 方式预测等待分位数；
  - `sti_transformer/`：STI-Transformer 训练、推理与评估代码。
- `src/selection/risk_choice.py`
  - 按风险偏好参数 `gamma` 对候选门店进行决策。
- `src/mfhr/k_controller.py`
  - 根据不确定性（如 `W90-W50`）计算优先权修正量 `K`。
- `src/control/congestion_correction.py`
  - 提供动态 `lambda` 修正逻辑。
- `src/metrics/`
  - 公平性（Jain 指标）、回归误差与统计工具函数。
- `experiments/`
  - 参数扫描、消融分析、结果导出、显著性检验等实验脚本。

---

## 3. 闭环执行链路（核心）

1. `main_experiment.py` 生成 `gamma_grid × lambda_grid × seeds` 任务列表；
2. 每个任务进入 `run_task`：
   - 调用 `run_closed_loop(config, gamma, lambda_, seed, ...)`；
   - 得到 `tau` 时间序列与公平性；
   - 计算稳态统计量（均值/方差/斜率）；
3. 汇总多随机种子结果，形成四张相图矩阵：
   - `tau_mean`、`tau_var`、`tau_slope`、`fairness`；
4. 将矩阵保存到 `phase_results.npz`，并绘制热图与四类案例曲线。

---

## 4. SimPy 系统机制（`simpy_system.py`）

### 4.1 到达过程

- 使用 NHPP（非齐次泊松过程）模拟时变到达率；
- 率函数 `nhpp_rate(t)` 为高斯峰型：基础流量 + 峰值扰动；
- 通过 thinning 方法生成到达时间。

### 4.2 单顾客决策流程

对每个到达事件：

1. 采样行程时间 `eta`；
2. 对每个候选门店预测等待分位数 (`W50`, `W90`)：
   - 有外部 predictor 时使用模型；
   - 否则用 MC 预测器；
3. 根据 `delta = W90-W50` 估计不确定性并计算 `K_j`；
4. 用 `choose_model(W50, W90, gamma)` 选择目标门店；
5. 构造优先级 `priority = arrival_time_at_store - K_j`；
6. 进入对应 `PriorityResource` 队列并开始服务；
7. 记录真实等待时间，最终聚合成 `tau` 序列和公平性。

### 4.3 输出指标

- `tau_series`：按分钟桶聚合的平均等待时间，含前后向补齐；
- `fairness`：基于所有顾客真实等待时间计算 Jain 指数。

---

## 5. 主要控制参数含义

- `gamma`：风险选择权重，影响“偏向低中位数等待”还是“规避高分位风险”；
- `lambda`：拥塞不确定性惩罚强度，影响 `K` 调整幅度；
- `K0/Kmax`：优先权基线与上限；
- `mc_samples`：MC 预测采样数，决定预测精度/耗时平衡；
- `burn_in`：用于统计稳态阶段时丢弃前期瞬态。

---

## 6. 代码阅读时观察到的工程特征

- 实验脚本与核心模块分层较清晰，便于替换预测器或策略函数；
- 多处保留了消融开关（`ABLATE_LAMBDA/FIX_GAMMA/NO_PRIORITY`），便于论文实验复现；
- 当前仓库中存在部分调试/历史代码残留（例如日志打印和大段注释旧实现），后续可考虑清理以提升可维护性。

---

## 7. 建议的阅读顺序

1. `experiments/main_experiment.py`（先把整体实验流程看清）；
2. `src/simulation/closed_loop.py`（参数如何注入系统）；
3. `src/simulation/simpy_system.py`（事件驱动核心逻辑）；
4. `src/selection/risk_choice.py` + `src/mfhr/k_controller.py`（策略核心）；
5. `src/prediction/quantile_mc.py` 与 `src/prediction/sti_transformer/`（预测器细节）；
6. `src/metrics/`（输出指标定义）。

