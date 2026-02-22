import numpy as np

d = np.load("results/raw_runs/gamma_lambda_phase.npz", allow_pickle=True)
gamma = d["gamma_grid"]; lam = d["lambda_grid"]
tau_mean = d["tau_mean"]; tau_var = d["tau_var"]; tau_slope = d["tau_slope"]

# rebuild labels (same as你的脚本参数)
delta = 0.15
s0 = 0.012
var_q = 0.75
tau_min = float(tau_mean.min())
q_var = float(np.quantile(tau_var, var_q))

high_tau = tau_mean > (1.0 + delta) * tau_min
high_drift = tau_slope > s0
label = np.zeros_like(tau_mean, dtype=int)
label[(~high_tau) & high_drift] = 1
label[high_tau & (~high_drift)] = 2
label[high_tau & high_drift] = 3
osc = tau_var > q_var

def pick(mask, key_mat, quant=0.5, prefer_low=True):
    idx = np.argwhere(mask)
    vals = key_mat[mask]
    order = np.argsort(vals) if prefer_low else np.argsort(-vals)
    idx = idx[order]
    k = int(len(idx) * quant)
    i, j = idx[min(k, len(idx)-1)]
    return int(i), int(j)

# 1) stable-optimal: label0, low tau_mean
i0,j0 = pick(label==0, tau_mean, quant=0.5, prefer_low=True)

# 2) drift-risk: label1, high slope
i1,j1 = pick(label==1, tau_slope, quant=0.5, prefer_low=False)

# 3) bad-high-risk: label3, high tau_mean
i3,j3 = pick(label==3, tau_mean, quant=0.5, prefer_low=False)

# 4) oscillatory: high var, but not the worst drift
osc_mask = osc & (tau_slope < np.quantile(tau_slope, 0.90))
io,jo = pick(osc_mask, tau_var, quant=0.5, prefer_low=False)

print("Stable-Optimal:", i0,j0,"gamma,lambda=",gamma[i0],lam[j0],"mean,var,slope=",tau_mean[i0,j0],tau_var[i0,j0],tau_slope[i0,j0])
print("Drift-Risk    :", i1,j1,"gamma,lambda=",gamma[i1],lam[j1],"mean,var,slope=",tau_mean[i1,j1],tau_var[i1,j1],tau_slope[i1,j1])
print("High-Risk     :", i3,j3,"gamma,lambda=",gamma[i3],lam[j3],"mean,var,slope=",tau_mean[i3,j3],tau_var[i3,j3],tau_slope[i3,j3])
print("Oscillatory   :", io,jo,"gamma,lambda=",gamma[io],lam[jo],"mean,var,slope=",tau_mean[io,jo],tau_var[io,jo],tau_slope[io,jo])