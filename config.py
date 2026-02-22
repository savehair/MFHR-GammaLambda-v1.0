import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 预测窗口
PRED_HORIZON = 30  # minutes

# γ–λ 参数范围
GAMMA_RANGE = (0.0, 1.0)
LAMBDA_RANGE = (0.0, 3.0)

# K 参数
K0 = 3.0
KMAX = 20.0

# 在线校准
ONLINE_LR = 0.01
UPDATE_INTERVAL = 5  # minutes