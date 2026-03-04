import torch
from torch.utils.data import Dataset
import numpy as np

class QueueDataset(Dataset):
    """
    输入:
        x_seq: [T, N, F]
    输出:
        y_future: 未来 horizon 的等待时间
    """

    def __init__(self, data, horizon=30):
        self.data = data
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.horizon

    def __getitem__(self, idx):
        x = self.data[idx]
        # 预测 horizon 后第 0 个门店的等待时间（标量）
        y = self.data[idx + self.horizon][0, 0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)