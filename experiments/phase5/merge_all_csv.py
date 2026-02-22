import pandas as pd
from pathlib import Path

ROOT = Path("results")

ablation = pd.read_csv(ROOT / "ablation_results.csv")
ablation["Experiment"] = "Ablation"

merged = ablation.copy()

merged.to_csv(ROOT / "paper_outputs" / "all_results.csv", index=False)

print("Merged CSV generated.")