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
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int, depth: int, mlp_ratio: int, dropout: float, act:str, max_seq_len: int, pre_norm: bool):
        super().__init__()

        # Project pose features to Transformer dimension
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)

        # Transformer Encoder
        self.transformer = TransformerEncoder(d_model=embed_dim, num_heads=num_heads, depth=depth, mlp_ratio=mlp_ratio, dropout=dropout, act=act, pre_norm=pre_norm)

        self.norm = nn.LayerNorm(embed_dim) if pre_norm else None

    def forward(self, x):
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
        x, _ = self.transformer(x)

        # Final normalization
        if self.norm is not None:
            x = self.norm(x)

        return x