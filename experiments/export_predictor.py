import torch
from src.prediction.sti_transformer.sti_model import STITransformer
from config import DEVICE

def load_predictor(adj):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)

    model = STITransformer(input_dim=4).to(DEVICE)
    model.load_state_dict(torch.load("results/best_model.pt", map_location=device))
    model.eval()

    from src.prediction.sti_transformer.predictor_interface import STIPredictor
    return STIPredictor(model, adj)