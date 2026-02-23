# MFHR-GammaLambda-v1.0

本项目用于运行 MFHR 闭环仿真（γ-λ 相图）、STI-Transformer 训练消融，并输出实验图表与汇总报告。

本文档面向“全新电脑（已安装 Anaconda，显卡 RTX 4060）”场景，提供从环境安装到测试、实验、数据导出、可视化的完整流程。

## 1. 环境安装（Anaconda + RTX 4060）

### 1.1 克隆仓库

```bash
git clone https://github.com/savehair/MFHR-GammaLambda-v1.0.git
cd MFHR-GammaLambda-v1.0
```

### 1.2 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate mfhr_sti
```

说明：
- `environment.yml` 已包含 `python=3.10`、`pytorch=2.2`、`pytorch-cuda=12.1`、`simpy`、`matplotlib` 等依赖。
- 若你本机 CUDA 驱动较新（RTX 4060 常见场景），该配置通常可直接使用。

### 1.3 快速确认 GPU 可用

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count())"
```

### 1.4 可选：限制线程（避免多进程时 CPU 过载）

Linux/macOS:
```bash
export OMP_NUM_THREADS=1
```

Windows PowerShell:
```powershell
$env:OMP_NUM_THREADS = "1"
```

## 2. 代码测试（先做冒烟测试）

项目根目录已提供冒烟脚本：

```bash
bash smoke_test.sh
```

该脚本会依次执行：
- 关键依赖检查（`numpy/matplotlib/tqdm/simpy`）
- 语法检查（`python -m compileall src`）
- 最小参数闭环实验（`experiments.main_experiment`）

如果你想指定解释器（例如特定 conda 环境）：

```bash
PYTHON_BIN=python bash smoke_test.sh
```

## 3. 运行实验

### 3.1 运行 γ-λ 主实验（闭环仿真）

建议先用小网格验证，再跑全量：

小规模（快速验证）：
```bash
python -m experiments.main_experiment \
  --gamma_points 5 \
  --lambda_points 5 \
  --seeds 2 \
  --n_workers 2 \
  --out_dir results
```

全量（论文级/正式结果）：
```bash
python -m experiments.main_experiment \
  --gamma_points 21 \
  --lambda_points 21 \
  --seeds 5 \
  --n_workers 8 \
  --out_dir results
```

主实验输出（默认在 `results/`）：
- `phase_results.npz`（核心数值结果）
- `heat_tau_mean.png`
- `heat_tau_var.png`
- `heat_tau_slope.png`
- `heat_fairness.png`
- `four_cases_tau.png`

### 3.2 运行 STI 模型消融实验

```bash
python -m experiments.run_ablation
```

输出：
- `results/ablation_results.csv`
- `results/best_model.pt`（训练过程中保存）

## 4. 数据导出

### 4.1 用内置 Phase5 脚本导出汇总文件

在已生成 `results/phase_results.npz` 和 `results/ablation_results.csv` 后执行：

```bash
python -m experiments.phase5.main_report_generator
```

输出目录：`results/paper_outputs/`
- `phase_summary.json`（相图统计摘要）
- `ablation_table.tex`（LaTeX 三线表）
- `experiment_section.md`（实验章节 Markdown 草稿）

### 4.2 可选：将 `phase_results.npz` 导出为 CSV

```bash
python - <<'PY'
import numpy as np
import pandas as pd

npz = np.load('results/phase_results.npz', allow_pickle=True)
gamma = npz['gamma_grid']
lam = npz['lambda_grid']

rows = []
for i, g in enumerate(gamma):
    for j, l in enumerate(lam):
        rows.append({
            'gamma': float(g),
            'lambda': float(l),
            'tau_mean': float(npz['tau_mean'][i, j]),
            'tau_var': float(npz['tau_var'][i, j]),
            'tau_slope': float(npz['tau_slope'][i, j]),
            'fairness': float(npz['fairness'][i, j]),
        })

pd.DataFrame(rows).to_csv('results/phase_results_flat.csv', index=False)
print('saved: results/phase_results_flat.csv')
PY
```

## 5. 数据可视化

### 5.1 主实验自动生成图

`experiments.main_experiment` 已自动生成四张热图和一张四场景曲线图：
- `results/heat_tau_mean.png`
- `results/heat_tau_var.png`
- `results/heat_tau_slope.png`
- `results/heat_fairness.png`
- `results/four_cases_tau.png`

### 5.2 代表性四场景 τ(t) 可视化（基于相图自动选点）

```bash
python -m experiments.plot_four_cases_tau \
  --npz results/phase_results.npz \
  --seeds 5 \
  --out_dir results/figures
```

输出：
- `results/figures/four_cases_tau_series.png`

## 6. 推荐的一键执行顺序

```bash
# 1) 激活环境
conda activate mfhr_sti

# 2) 冒烟测试
bash smoke_test.sh

# 3) 主实验（可先小规模）
python -m experiments.main_experiment --gamma_points 5 --lambda_points 5 --seeds 2 --n_workers 2 --out_dir results

# 4) 消融实验
python -m experiments.run_ablation

# 5) 导出报告文件
python -m experiments.phase5.main_report_generator

# 6) 代表性可视化
python -m experiments.plot_four_cases_tau --npz results/phase_results.npz --seeds 5 --out_dir results/figures
```

## 7. 常见问题

### 7.1 `ModuleNotFoundError: simpy`
说明当前 Python 环境不是 `mfhr_sti` 或依赖未安装完整。

处理：
```bash
conda activate mfhr_sti
python -c "import simpy; print(simpy.__version__)"
```

### 7.2 `ModuleNotFoundError: src`
请在项目根目录运行，并优先使用模块方式：
```bash
python -m experiments.main_experiment ...
```

### 7.3 Matplotlib 缓存目录不可写
可指定：
```bash
export MPLCONFIGDIR=.cache/matplotlib
```

## 8. 结果目录总览

典型输出结构如下：

```text
results/
├── ablation_results.csv
├── best_model.pt
├── phase_results.npz
├── heat_tau_mean.png
├── heat_tau_var.png
├── heat_tau_slope.png
├── heat_fairness.png
├── four_cases_tau.png
├── figures/
│   └── four_cases_tau_series.png
└── paper_outputs/
    ├── phase_summary.json
    ├── ablation_table.tex
    └── experiment_section.md
```
