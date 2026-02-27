import torch
torch.set_num_threads(1)
from src.prediction.sti_transformer.train import train_model
from src.prediction.sti_transformer.evaluate import evaluate
from src.prediction.sti_transformer.ablation import build_model
from src.prediction.sti_transformer.dataset import QueueDataset
from config import DEVICE

def run_training(data_train, data_val, config, adj):

    dataset_train = QueueDataset(data_train)
    dataset_val = QueueDataset(data_val)

    model = build_model(config).to(DEVICE)

    best_val = 1e9
    patience = 5
    counter = 0

    for epoch in range(50):

        model = train_model(model, dataset_train, adj, epochs=1)
        metrics = evaluate(model, dataset_val, adj)

        val_loss = metrics["MAE"]

        print(f"Epoch {epoch} | Val MAE {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "results/best_model.pt")
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered.")
            break

    try:
        state_dict = torch.load("results/best_model.pt", map_location=DEVICE, weights_only=True)
    except TypeError:
        state_dict = torch.load("results/best_model.pt", map_location=DEVICE)

    model.load_state_dict(state_dict)
    return model
