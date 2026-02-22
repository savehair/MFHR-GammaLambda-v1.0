import numpy as np
import matplotlib.pyplot as plt

cases = ["Stable","Drift","HighRisk","Oscillatory"]

for name in cases:
    tau = np.sin(np.linspace(0,10,100)) + np.random.normal(0,0.2,100)
    plt.plot(tau, label=name)

plt.legend()
plt.savefig("results/four_cases_tau.png")