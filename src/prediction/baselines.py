from __future__ import annotations

import torch
import torch.nn as nn


torch.set_num_threads(1)


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Input must be [B, H, N, F].")

        batch, history, nodes, feat = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(batch * nodes, history, feat)
        out, _ = self.rnn(seq)
        pred = self.head(out[:, -1, :])
        return pred.reshape(batch, nodes, 2)


class GRUModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Input must be [B, H, N, F].")

        batch, history, nodes, feat = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(batch * nodes, history, feat)
        out, _ = self.rnn(seq)
        pred = self.head(out[:, -1, :])
        return pred.reshape(batch, nodes, 2)


class PureTransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Input must be [B, H, N, F].")

        batch, history, nodes, feat = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(batch * nodes, history, feat)
        enc = self.input_proj(seq)
        enc = self.encoder(enc)
        pred = self.head(enc[:, -1, :])
        return pred.reshape(batch, nodes, 2)


def _shape_assertion() -> None:
    x = torch.randn(2, 60, 15, 4)
    lstm = LSTMModel(input_dim=4)
    gru = GRUModel(input_dim=4)
    trans = PureTransformerModel(input_dim=4)

    assert lstm(x).shape == (2, 15, 2)
    assert gru(x).shape == (2, 15, 2)
    assert trans(x).shape == (2, 15, 2)


# pytest assertion example

def test_baseline_output_shapes() -> None:
    _shape_assertion()


if __name__ == "__main__":
    _shape_assertion()
