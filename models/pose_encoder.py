import torch
import torch.nn as nn

from .utils import MLP
from models.transformer import TransformerEncoder


class PoseEncoder(nn.Module):
    """
    Transformer-based encoder for pose sequences.

    Input:
        x: (B, T, input_dim)

    Output:
        x: (B, T, d_model)
    """
    def __init__(self, input_dim: int, embed_ratio: int, num_heads: int, depth: int, mlp_ratio: int, dropout: float, act:str, max_seq_len: int):
        super().__init__()

        d_model = int(embed_ratio * input_dim)

        # Project pose features to Transformer dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer Encoder
        self.transformer = TransformerEncoder(d_model=d_model, num_heads=num_heads, depth=depth, mlp_ratio=mlp_ratio, dropout=dropout, act=act)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, pre_norm: bool = False):
        """
        Parameters
        ----------
        x : torch.Tensor
            Pose sequence.
            Shape: (B, T, input_dim)

        mask : torch.Tensor, optional
            Padding mask.
            Shape: (B, T)
            True = ignored position.

        Returns
        -------
        torch.Tensor
            Shape: (B, T, d_model)
        """

        B, T, _ = x.shape

        if T > self.pos_embedding.size(1):
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"max_seq_len {self.pos_embedding.size(1)}"
            )

        # (B, T, input_dim) -> (B, T, d_model)
        x = self.input_proj(x)

        # Add positional information
        x = x + self.pos_embedding[:, :T, :]

        # Transformer
        x, _ = self.transformer(x, pre_norm=pre_norm)

        # Final normalization
        if pre_norm:
            x = self.norm(x)

        return x