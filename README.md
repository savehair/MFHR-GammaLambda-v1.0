# MFHR γ–λ 相图（可复现工程包）

严格对齐论文口径：
- 主变量：K（分钟）
- 禁止改道（No rerouting）
- 唯一主指标：τ = Min-Sum AWT
- 预测输出语义：P50/P90 为“用户到店时等待时间预测”

## 1) 环境安装（Conda）
在仓库根目录执行：

```bash
conda env create -f environment.yml -p ./envs/mfhr-sti
conda activate ./envs/mfhr-sti
```

## 2) 运行 γ–λ 相图
```bash
python experiments/run_gamma_lambda_phase.py
```

输出：
- results/raw_runs/gamma_lambda_phase.npz

## 3) PyCharm 绑定解释器
Settings/Preferences → Project → Python Interpreter → Add Interpreter → Conda → Existing environment  
选择 ./envs/mfhr-env/bin/python（Windows 选择 ./envs/mfhr-env/python.exe）

## 4) 说明
src/simulation/closed_loop.py 的系统动态为“占位模拟”，用于保证 γ–λ sweep 与统计流程可运行。
你可以替换为 SimPy 的 M/M/s 离散事件仿真，但不要改变四步闭环结构与 τ 定义。
