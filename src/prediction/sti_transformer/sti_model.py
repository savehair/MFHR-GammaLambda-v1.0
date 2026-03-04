import torch
import torch.nn as nn
from .nif_module import NIFModule
from .gcn_module import GCNLayer
from .transformer_module import TimeTransformer

class STITransformer(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim=64,
                 use_nif=True,
                 use_gcn=True,
                 use_transformer=True):

        super().__init__()

        self.use_nif = use_nif
        self.use_gcn = use_gcn
        self.use_transformer = use_transformer

        # 当关闭 NIF 时，仍需把输入维度映射到 hidden_dim，
        # 以兼容后续 GCN/Transformer/Head 的固定 hidden 维度。
        self.input_proj = (
            nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        )

        if use_nif:
            self.nif = NIFModule(input_dim, hidden_dim)

        if use_gcn:
            self.gcn = GCNLayer(hidden_dim, hidden_dim)

        if use_transformer:
            self.transformer = TimeTransformer(hidden_dim)

        # 输出 P50 和 P90
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x, adj=None):

        if self.use_nif:
            x = self.nif(x)
        else:
            x = self.input_proj(x)

        if self.use_gcn and adj is not None:
            x = self.gcn(x, adj)

        if self.use_transformer:
            x = self.transformer(x)

        out = self.head(x[:, -1, :])
        return out  # [batch, 2]
