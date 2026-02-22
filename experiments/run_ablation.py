import os
import pandas as pd
import torch
torch.set_num_threads(1)
import numpy as np
from experiments.synthetic_data import generate_synthetic_dataset
from experiments.train_runner import run_training
from src.prediction.sti_transformer.evaluate import evaluate
from config import DEVICE

os.makedirs("results", exist_ok=True)

train_data, val_data, test_data = generate_synthetic_dataset()

adj = torch.eye(train_data.shape[1]).to(DEVICE)

configs = [
    {"name": "A_Full", "use_nif": True, "use_gcn": True, "use_transformer": True, "input_dim":4},
    {"name": "B_NoNIF", "use_nif": False, "use_gcn": True, "use_transformer": True, "input_dim":4},
    {"name": "C_NoGCN", "use_nif": True, "use_gcn": False, "use_transformer": True, "input_dim":4},
    {"name": "D_NoTransformer", "use_nif": True, "use_gcn": True, "use_transformer": False, "input_dim":4},
]

results = []

for cfg in configs:
    print("Running:", cfg["name"])

    model = run_training(train_data, val_data, cfg, adj)

    metrics = evaluate(model, test_data, adj)
    metrics["Group"] = cfg["name"]

    results.append(metrics)

df = pd.DataFrame(results)
df.to_csv("results/ablation_results.csv", index=False)

print(df)