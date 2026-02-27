import torch
import numpy as np
from config import DEVICE
from src.metrics.regression import mae, rmse, mape

def evaluate(model, dataset, adj):

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