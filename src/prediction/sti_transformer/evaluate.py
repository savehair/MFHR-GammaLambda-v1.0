import torch
import numpy as np
from config import DEVICE
from src.metrics.regression import mae, rmse, mape
from src.prediction.sti_transformer.dataset import QueueDataset

def evaluate(model, dataset, adj):

    if isinstance(dataset, np.ndarray):
        dataset = QueueDataset(dataset)

    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for x, y in dataset:
            x = x.unsqueeze(0).to(DEVICE)
            out = model(x, adj)
            preds.append(out[0,0].item())  # P50
            trues.append(float(y.item() if y.ndim == 0 else y.reshape(-1)[0].item()))

    preds = np.array(preds)
    trues = np.array(trues)

    return {
        "MAE": mae(trues, preds),
        "RMSE": rmse(trues, preds),
        "MAPE": mape(trues, preds)
    }
