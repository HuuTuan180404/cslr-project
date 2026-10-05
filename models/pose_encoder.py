import torch
import torch.nn as nn

from models.mlp import MLP
from models.transformer import TransformerEncoder
from models.positional_encoding import PositionalEncoding


class PoseEncoder(nn.Module):
    """
    Transformer-based encoder for pose sequences.

    Input:
        x: (B, T, input_dim)

    Output:
        x: (B, T, d_model)
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        depth,
        mlp_ratio,
        dropout,
        act,
        max_frames,
        pre_norm,
        pe_type,
    ):
        super().__init__()
        # Learnable positional embedding
        self.pe = PositionalEncoding(
            pe_type=pe_type, d_model=embed_dim, max_len=max_frames
        )

        # Transformer Encoder
        self.transformer = TransformerEncoder(
            d_model=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            act=act,
            pre_norm=pre_norm,
        )

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

        # Add positional information
        x = self.pe(x)

        # Transformer
        x, _ = self.transformer(x)

        # Final normalization
        if self.norm is not None:
            x = self.norm(x)

        return x
