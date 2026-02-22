import torch
from config import DEVICE

class STIPredictor:

    def __init__(self, model, adj):
        self.model = model
        self.adj = adj
        self.model.eval()

    def predict_quantiles(self, features):
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            out = self.model(x, self.adj)
            p50 = out[0,0].item()
            p90 = out[0,1].item()
        return p50, p90