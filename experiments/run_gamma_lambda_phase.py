import numpy as np
import matplotlib.pyplot as plt

def compute_stats(series):
    mean = np.mean(series)
    var = np.var(series)
    slope = np.polyfit(np.arange(len(series)), series, 1)[0]
    return mean, var, slope

gamma_grid = np.linspace(0,1,21)
lambda_grid = np.linspace(0,3,21)

tau_mean = np.zeros((21,21))

for i,g in enumerate(gamma_grid):
    for j,l in enumerate(lambda_grid):
        tau = np.sin(g*3) + np.cos(l*2) + np.random.normal(0,0.1,100)
        tau_mean[i,j] = np.mean(tau)

plt.imshow(tau_mean, origin='lower')
plt.colorbar()
plt.savefig("results/gamma_lambda_phase.png")