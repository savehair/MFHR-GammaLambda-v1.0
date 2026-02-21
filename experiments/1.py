# import numpy as np
#
# d = np.load("results/raw_runs/gamma_lambda_phase.npz", allow_pickle=True)
# #
# # print(d.files)
# #
# # for k in d.files:
# #     if hasattr(d[k], "shape"):
# #         print(k, d[k].shape)
#
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(6,5))
# plt.imshow(d["tau_mean"], origin="lower", aspect="auto")
# plt.colorbar(label="tau (Min-Sum AWT)")
# plt.xlabel("lambda index")
# plt.ylabel("gamma index")
# plt.title("Gamma-Lambda Phase Diagram (tau)")
# plt.show()
import simpy
