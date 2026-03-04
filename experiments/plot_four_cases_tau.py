import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.simulation.closed_loop import run_closed_loop


def pick_representatives(npz_path, delta=0.15, s0=0.012, var_q=0.75):
    d = np.load(npz_path, allow_pickle=True)
    gamma = d["gamma_grid"]
    lam = d["lambda_grid"]
    tau_mean = d["tau_mean"]
    tau_var = d["tau_var"]
    tau_slope = d["tau_slope"]

    tau_min = float(tau_mean.min())
    q_var = float(np.quantile(tau_var, var_q))

    high_tau = tau_mean > (1.0 + delta) * tau_min
    high_drift = tau_slope > s0

    label = np.zeros_like(tau_mean, dtype=int)
    label[(~high_tau) & high_drift] = 1
    label[high_tau & (~high_drift)] = 2
    label[high_tau & high_drift] = 3

    osc_flag = tau_var > q_var

    def pick(mask, key_mat, prefer_low=True):
        idx = np.argwhere(mask)
        vals = key_mat[mask]
        order = np.argsort(vals) if prefer_low else np.argsort(-vals)
        idx = idx[order]
        i, j = idx[len(idx)//2]
        return int(i), int(j)

    # 1 Stable
    i0, j0 = pick(label == 0, tau_mean, prefer_low=True)

    # 2 Drift
    i1, j1 = pick(label == 1, tau_slope, prefer_low=False)

    # 3 High Risk
    i3, j3 = pick(label == 3, tau_mean, prefer_low=False)

    # 4 Oscillatory
    osc_mask = osc_flag & (tau_slope < np.quantile(tau_slope, 0.90))
    io, jo = pick(osc_mask, tau_var, prefer_low=False)

    reps = [
        ("Stable-Optimal", gamma[i0], lam[j0]),
        ("Drift-Risk", gamma[i1], lam[j1]),
        ("High-Risk", gamma[i3], lam[j3]),
        ("Oscillatory", gamma[io], lam[jo]),
    ]

    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="results/figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reps = pick_representatives(args.npz)

    # 读取 config
    d = np.load(args.npz, allow_pickle=True)
    config = eval(d["config"].item())

    plt.figure(figsize=(10, 6))

    for name, gamma, lam in reps:
        all_series = []

        for s in range(args.seeds):
            seed = 10000 + s
            series, _ = run_closed_loop(config, gamma=float(gamma), lambda_=float(lam), seed=seed)
            all_series.append(series)

        all_series = np.array(all_series)
        mean_series = np.mean(all_series, axis=0)
        std_series = np.std(all_series, axis=0)

        t = np.arange(len(mean_series))

        plt.plot(t, mean_series, label=f"{name} (γ={gamma:.2f}, λ={lam:.2f})")
        plt.fill_between(t, mean_series - std_series, mean_series + std_series, alpha=0.2)

    plt.xlabel("Time (minute)")
    plt.ylabel("τ(t)  (Min-Sum AWT)")
    plt.title("Four Representative Phase Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "four_cases_tau_series.png", dpi=200)
    plt.close()

    print("[OK] Saved:", out_dir / "four_cases_tau_series.png")


if __name__ == "__main__":
    main()