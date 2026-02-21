# MFHR γ–λ 相图实验报告（整合版）

## 1. 实验目标
在闭环 MFHR 框架下，仅开展 γ–λ 相图分析，研究用户风险偏好 γ 与系统鲁棒放大系数 λ 的耦合效应。

- 主定义：K
- 扩展版本：B（本实验不启用）
- 禁止改道（No rerouting）
- 唯一主指标：τ（Min-Sum AWT）

## 2. 闭环四阶段定义（严格口径）

### Step 1：预测模块输出
输出到店时等待分位：
( W_hat_P50, W_hat_P90 )

### Step 2：用户选择模型（风险参数 γ）
j* = argmin_j [ W_hat_P50^j + γ ( W_hat_P90^j - W_hat_P50^j ) ]
- γ=0：risk-neutral
- γ>0：risk-averse

### Step 3：MFHR 风险调参（映射到 K）
K_j = K0 + λ ( W_hat_P90^j - W_hat_P50^j ),  with K_j <= Kmax

### Step 4：反馈更新（闭环）
theta_{t+1} = theta_t - η ∇L(ε)
（工程中保留接口，训练实现可替换。）

## 3. 固定实验设定（单位：分钟）
- 门店数 N=15
- 仿真时长 120 min（实现中离散为 T=240 步，占位 dt=0.5min）
- 到达过程：NHPP（默认配置在 configs/default.yaml）
- 服务参数：μ=1.2 users/min，servers=3
- K0=3, Kmax=15
- γ∈[0,1]，λ∈[0,2]（21×21 网格）

## 4. 评价指标（唯一主指标）
τ = mean(real waiting times)

占位实现以预测中位等待作为 τ 近似；替换为离散事件仿真后，使用真实等待即可保持口径一致。

## 5. 相区判定准则
每个 (γ,λ) 运行多随机种子，记录 τ 时间序列，丢弃 burn-in 前 20% 后计算：
- τ_mean
- τ_var
- slope（τ 随时间线性拟合斜率）

典型相区：
- 稳定区：低 τ_mean + 低 τ_var + |slope|≈0
- 过度保守区：τ_mean 上升但 τ_var 低
- 震荡区：τ_var 高但 slope≈0
- 发散区：slope>0（或 τ 无界上升）

## 6. 可复现性
- 固定 seed 列表
- 固定网格范围与点数
- 固定 K0/Kmax 与其他系统参数
- environment.yml 提供精确版本
