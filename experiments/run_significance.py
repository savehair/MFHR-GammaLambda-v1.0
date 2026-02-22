import pandas as pd
from src.metrics.statistics import t_test

df = pd.read_csv("results/ablation_results.csv")

full = df[df["Group"]=="A_Full"]["MAE"].values
others = df[df["Group"]!="A_Full"]["MAE"].values

p_value = t_test(full, others)

print("p-value:", p_value)