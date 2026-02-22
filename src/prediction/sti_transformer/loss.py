import torch

def pinball_loss(pred, target, quantile):
    diff = target - pred
    return torch.mean(torch.maximum(
        quantile * diff,
        (quantile - 1) * diff
    ))