import torch
from torch.utils.data import DataLoader
from .loss import pinball_loss
from config import DEVICE

def train_model(model, dataset, adj, epochs=20, lr=1e-3):

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).reshape(-1)

            optimizer.zero_grad()
            out = model(x, adj)

            loss = (
                pinball_loss(out[:,0], y, 0.5) +
                pinball_loss(out[:,1], y, 0.9)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss {total_loss/len(loader):.4f}")

    return model