import torch.nn as nn
from utils.utils import get_activation


class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio, act: str, dropout: float):
        super().__init__()

        hidden_dim = mlp_ratio * d_model

        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.activation = get_activation(act)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
