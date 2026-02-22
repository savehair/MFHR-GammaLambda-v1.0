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
        y = self.data[idx + self.horizon][:, 0]  # 假设第0维为等待时间
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32)