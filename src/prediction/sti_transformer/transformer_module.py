import torch.nn as nn

class TimeTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x):
        return self.encoder(x)