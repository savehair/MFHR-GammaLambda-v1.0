from pathlib import Path
import shutil

SRC = Path("results")
DST = SRC / "paper_outputs" / "figures"
DST.mkdir(parents=True, exist_ok=True)

files = [
    "heat_tau_mean.png",
    "heat_tau_var.png",
    "heat_tau_slope.png",
    "heat_fairness.png",
    "four_cases_tau.png",
]

for f in files:
    if (SRC / f).exists():
        shutil.copy(SRC / f, DST / f)

print("Figures organized.")