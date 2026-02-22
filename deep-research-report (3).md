# 项目安装和运行指南

下面给出在空白环境下从头配置并运行实验的步骤，包括代码克隆、环境创建、依赖安装和命令示例。

## 环境准备

1. **克隆代码仓库**

   ```bash
   git clone <your_repo_url>
   cd <repo_folder>
   ```

2. **创建并激活 Conda 环境**

   项目根目录下有 `environment.yml`，执行：

   ```bash
   conda env create -f environment.yml
   conda activate mfhr_sti  # 使用 environment.yml 中的 name
   ```

3. **安装 PyTorch（CPU/GPU）**

   - 如果使用 **GPU** (RTX 4060，CUDA 11.8)：
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
   - 如果只用 **CPU**：
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     ```
   （此处基于 PyTorch 官方索引）

4. **设置并行线程**

   建议设置环境变量以限制 OpenMP 线程数：
   ```bash
   export OMP_NUM_THREADS=4   # 示例：将 CPU 线程数设为 4
   ```

## 实验运行步骤

1. **生成合成数据**

   ```bash
   python src/data/leak_free_synth.py
   ```

   这会生成 `data/synth.npz`，用于后续训练和评估。

2. **训练 STI-Transformer**

   ```bash
   python src/train/train_forecaster.py --config config.yaml
   ```

   - `config.yaml` 包含超参数（学习率、批量大小等）和路径设置。  
   - 输出模型文件：`results/model_best.pt`，以及 `results/metrics.json` （验证 MAE/RMSE）。

3. **消融与预测评估**

   - **消融实验**（关闭 NIF/GCN 等）：
     ```bash
     python src/train/train_forecaster.py --config config_ablation.yaml
     ```
     结果保存到 `results/ablation_results.csv` 中，记录各组 MAE/RMSE。
   - **准确率评估**：
     ```bash
     python src/prediction/sti_transformer/evaluate.py --model results/model_best.pt --npz data/synth.npz
     ```
     输出 `results/accuracy.json` 包含 MAE/RMSE/wMAPE/覆盖率/宽度 等。

4. **γ–λ 闭环仿真**

   ```bash
   python experiments/main_experiment.py --n_workers 8
   ```

   该命令并行扫描 γ、λ 网格并保存结果到 `results/phase_results.npz`；同时生成热图和 τ 时间曲线。  
   - 推荐使用 `--n_workers` 设为 CPU 核心数（如 8）。  
   - Windows 系统下需确保 `if __name__ == "__main__"` 存在。

5. **后处理与报告**

   ```bash
   python experiments/phase5/main_report_generator.py
   ```
   生成 LaTeX 表格 (`ablation_table.tex`) 和 Markdown 结果章节 (`experiment_section.md`)。

## 结果输出格式

- **CSV/JSON 格式**：例如 `ablation_results.csv`、`accuracy.json`、`phase_summary.json` 等。  
- **图像文件**：热图（`heat_tau_mean.png` 等）、τ(t) 曲线图（`four_cases_tau.png`）保存在 `results/` 目录下。  
- **三线表**：存为 LaTeX (`results/ablation_table.tex`) 便于论文引用。

## GitHub 同步

```bash
git add .
git commit -m "实验结果"
git push origin main
```

## 验证安装

- 使用 **pytest** 运行单元测试（需要预先生成合成数据）：
  ```bash
  pytest -q
  ```

完成以上步骤后，你的环境即配置完毕，并能复现相关实验结果。