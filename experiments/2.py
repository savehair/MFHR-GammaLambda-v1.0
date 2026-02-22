import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default="results/raw_runs/gamma_lambda_phase.npz")
    ap.add_argument("--out_dir", type=str, default="results/figures")
    ap.add_argument("--delta", type=float, default=0.15)   # conservative threshold
    ap.add_argument("--s0", type=float, default=0.012)      # drift threshold (tune)
    ap.add_argument("--var_q", type=float, default=0.75)    # oscillation quantile
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.npz, allow_pickle=True)
    gamma = d["gamma_grid"]
    lam = d["lambda_grid"]
    tau_mean = d["tau_mean"]
    tau_var = d["tau_var"]
    tau_slope = d["tau_slope"]

    def save_heat(mat, title, fname, cbar):
        plt.figure(figsize=(7,5))
        plt.imshow(mat, origin="lower", aspect="auto")
        plt.colorbar(label=cbar)
        plt.xlabel("lambda index")
        plt.ylabel("gamma index")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()

    save_heat(tau_mean, "Gamma-Lambda Phase (tau_mean)", "heat_tau_mean.png", "tau_mean")
    save_heat(tau_var, "Gamma-Lambda Phase (tau_var)", "heat_tau_var.png", "tau_var")
    save_heat(tau_slope, "Gamma-Lambda Phase (tau_slope)", "heat_tau_slope.png", "tau_slope")

    # ----- 4-region labeling -----
    tau_min = float(tau_mean.min())
    q_var = float(np.quantile(tau_var, args.var_q))

    # labels: 0=lowTau-lowDrift, 1=lowTau-highDrift, 2=highTau-lowDrift, 3=highTau-highDrift
    high_tau = tau_mean > (1.0 + args.delta) * tau_min
    high_drift = tau_slope > args.s0

    label = np.zeros_like(tau_mean, dtype=int)
    label[(~high_tau) & high_drift] = 1
    label[high_tau & (~high_drift)] = 2
    label[high_tau & high_drift] = 3

    # (optional) mark high-variance points as "oscillatory flag"
    osc_flag = tau_var > q_var

    plt.figure(figsize=(7,5))
    plt.imshow(label, origin="lower", aspect="auto")
    plt.colorbar(label="phase label (0-3)")
    plt.xlabel("lambda index")
    plt.ylabel("gamma index")
    plt.title("Phase Labels (tau_mean & tau_slope)")
    plt.tight_layout()
    plt.savefig(out_dir / "phase_labels.png", dpi=200)
    plt.close()

    # print counts
    uniq, cnt = np.unique(label, return_counts=True)
    print("Label counts:", dict(zip(uniq.tolist(), cnt.tolist())))
    print(f"Oscillation-flag count (tau_var>{args.var_q}q):", int(np.sum(osc_flag)))

if __name__ == "__main__":
    main()