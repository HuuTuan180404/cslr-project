import torch
import torch.nn as nn
from typing import Tuple

from visual_encoder import VisualEncoder
from .utils import MLP

class CrossAttention(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, num_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__()

        d_model = (x_dim + y_dim) // 2

        if d_model % num_heads != 0:
            raise ValueError(   
                f"d_model ({d_model}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.x_proj = nn.Identity() if x_dim == d_model else nn.Linear(x_dim, d_model)
        self.y_proj = nn.Identity() if y_dim == d_model else nn.Linear(y_dim, d_model)

        self.x_to_y_cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.y_to_x_cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.x_proj_out = nn.Identity() if d_model == x_dim else nn.Linear(d_model, x_dim)
        self.y_proj_out = nn.Identity() if d_model == y_dim else nn.Linear(d_model, y_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, Tx, x_dim)
        # y: (B, Ty, y_dim)

        x_proj = self.x_proj(x)
        y_proj = self.y_proj(y)

        x_cross, _ = self.x_to_y_cross_attn(x_proj, y_proj, y_proj)
        y_cross, _ = self.y_to_x_cross_attn(y_proj, x_proj, x_proj)

        x_out = self.x_proj_out(x_cross)
        y_out = self.y_proj_out(y_cross)

        return x_out, y_out


class HandEncoderLayer(nn.Module):
    def __init__(self, pose_d_model, pose_num_heads, rgb_embed_dim: int = 768, act: str =  "gelu", mlp_ratio=4, dropout: float = 0.0, pose_rgb_num_heads: int = 15, pose_pose_num_heads: int = 15):
        super().__init__()

        # ============================================================
        # Pose <-> RGB Cross Attention
        # ============================================================
        self.lh_rgb_pose_cross = CrossAttention(x_dim=pose_d_model, y_dim=rgb_embed_dim, num_heads=pose_rgb_num_heads, dropout=dropout)
        self.rh_rgb_pose_cross = CrossAttention(x_dim=pose_d_model, y_dim=rgb_embed_dim, num_heads=pose_rgb_num_heads, dropout=dropout)

        # ============================================================
        # 1. Pose Self-Attention
        # ============================================================

        # Left hand
        self.lh_norm1 = nn.LayerNorm(pose_d_model)

        self.lh_attn = nn.MultiheadAttention(embed_dim=pose_d_model, num_heads=pose_num_heads, dropout=dropout, batch_first=True)

        self.lh_attn_dropout = nn.Dropout(dropout)

        # Right hand
        self.rh_norm1 = nn.LayerNorm(pose_d_model)

        self.rh_attn = nn.MultiheadAttention(embed_dim=pose_d_model, num_heads=pose_num_heads, dropout=dropout, batch_first=True)

        self.rh_attn_dropout = nn.Dropout(dropout)

        # ============================================================
        # 2. Left Pose <-> Right Pose Cross Attention
        # ============================================================

        self.lh_pose_rh_pose_attn = CrossAttention(x_dim=pose_d_model, y_dim=pose_d_model, num_heads=pose_pose_num_heads, dropout=dropout)

        # Norm 2
        self.lh_norm2 = nn.LayerNorm(pose_d_model)
        self.rh_norm2 = nn.LayerNorm(pose_d_model)

        # Dropout for cross attention
        self.lh_cross_dropout = nn.Dropout(dropout)
        self.rh_cross_dropout = nn.Dropout(dropout)

        # ============================================================
        # 3. Feed Forward Network
        # ============================================================

        self.lh_mlp = MLP(d_model=pose_d_model, mlp_ratio=mlp_ratio, act=act, dropout=dropout)

        self.rh_mlp = MLP(d_model=pose_d_model, mlp_ratio=mlp_ratio, act=act, dropout=dropout)

        # Norm 3
        self.lh_norm3 = nn.LayerNorm(pose_d_model)
        self.rh_norm3 = nn.LayerNorm(pose_d_model)

        # Dropout
        self.lh_mlp_dropout = nn.Dropout(dropout)
        self.rh_mlp_dropout = nn.Dropout(dropout)

    def forward(self, lh_pose: torch.Tensor, rh_pose: torch.Tensor, lh_rgb: torch.Tensor, rh_rgb: torch.Tensor, pre_norm: bool = False):

        # ============================================================
        # Pose <-> RGB Cross Attention
        # ============================================================
        lh_pose_rgb_attn, lh_rgb = self.lh_rgb_pose_cross(lh_pose, lh_rgb)
        rh_pose_rgb_attn, rh_rgb = self.rh_rgb_pose_cross(rh_pose, rh_rgb)

        if pre_norm: # PRE-NORM
            # ========================================================
            # 1. Pose Self-Attention
            # ========================================================
            # LH
            lh_residual = lh_pose_rgb_attn

            lh_norm = self.lh_norm1(lh_pose_rgb_attn)

            lh_attn_out, _ = self.lh_attn(lh_norm, lh_norm, lh_norm)

            lh_attn_out = self.lh_attn_dropout(lh_attn_out)

            lh_attn_out = lh_residual + lh_attn_out

            # RH
            rh_residual = rh_pose_rgb_attn

            rh_norm = self.rh_norm1(rh_pose_rgb_attn)

            rh_attn_out, _ = self.rh_attn(rh_norm, rh_norm, rh_norm)

            rh_attn_out = self.rh_attn_dropout(rh_attn_out)

            rh_attn_out = rh_residual + rh_attn_out

            # ========================================================
            # 2. LH <-> RH Cross Attention
            # ========================================================

            lh_residual = lh_attn_out
            rh_residual = rh_attn_out

            lh_cross_out, rh_cross_out = self.lh_pose_rh_pose_attn(lh_attn_out, rh_attn_out)

            lh_cross_out = self.lh_cross_dropout(lh_cross_out)
            rh_cross_out = self.rh_cross_dropout(rh_cross_out)

            lh_attn_out = lh_residual + lh_cross_out
            rh_attn_out = rh_residual + rh_cross_out

            # ========================================================
            # 3. MLP
            # ========================================================

            lh_residual = lh_attn_out
            rh_residual = rh_attn_out

            lh_mlp_out = self.lh_mlp(self.lh_norm3(lh_attn_out))

            rh_mlp_out = self.rh_mlp(self.rh_norm3(rh_attn_out))

            lh_mlp_out = self.lh_mlp_dropout(lh_mlp_out)
            rh_mlp_out = self.rh_mlp_dropout(rh_mlp_out)

            lh_mlp_out = lh_residual + lh_mlp_out
            rh_mlp_out = rh_residual + rh_mlp_out

        else: # POST-NORM

            # ========================================================
            # 1. Pose Self-Attention
            # ========================================================

            # LH
            lh_attn_out, _ = self.lh_attn(lh_pose_rgb_attn, lh_pose_rgb_attn, lh_pose_rgb_attn)
            lh_attn_out = self.lh_attn_dropout(lh_attn_out)
            lh_attn_out = self.lh_norm1(lh_pose_rgb_attn + lh_attn_out)

            # RH
            rh_attn_out, _ = self.rh_attn(rh_pose_rgb_attn, rh_pose_rgb_attn, rh_pose_rgb_attn)
            rh_attn_out = self.rh_attn_dropout(rh_attn_out)
            rh_attn_out = self.rh_norm1(rh_pose_rgb_attn + rh_attn_out)

            # ========================================================
            # 2. LH <-> RH Cross Attention
            # ========================================================

            lh_cross_out, rh_cross_out = self.lh_pose_rh_pose_attn(lh_attn_out, rh_attn_out)

            lh_cross_out = self.lh_cross_dropout(lh_cross_out)
            rh_cross_out = self.rh_cross_dropout(rh_cross_out)

            lh_attn_out = self.lh_norm2(lh_attn_out + lh_cross_out)

            rh_attn_out = self.rh_norm2(rh_attn_out + rh_cross_out)

            # ========================================================
            # 3. MLP
            # ========================================================

            lh_mlp_out = self.lh_mlp(lh_attn_out)
            rh_mlp_out = self.rh_mlp(rh_attn_out)

            lh_mlp_out = self.lh_mlp_dropout(lh_mlp_out)
            rh_mlp_out = self.rh_mlp_dropout(rh_mlp_out)

            lh_mlp_out = self.lh_norm3(lh_attn_out + lh_mlp_out)

            rh_mlp_out = self.rh_norm3(rh_attn_out + rh_mlp_out)

        return lh_mlp_out, rh_mlp_out


class HandEncoder(nn.Module):
    def __init__(self, d_model, num_heads, rgb_image_size: int = 112, rgb_patch_size: int = 16, rgb_in_channels: int = 3, rgb_embed_dim: int = 768, act: str = "gelu", rgb_num_heads: int = 12, depth=4, dropout=0.1, pose_rgb_num_heads: int = 15, pose_pose_num_heads: int = 15, mlp_ratio:int = 4):

        super().__init__()
        self.layers = nn.ModuleList([
            HandEncoderLayer(pose_d_model=d_model, pose_num_heads=num_heads, rgb_embed_dim=rgb_embed_dim, act=act, mlp_ratio=mlp_ratio, dropout=dropout, pose_rgb_num_heads=pose_rgb_num_heads, pose_pose_num_heads=pose_pose_num_heads)
            for _ in range(depth)
        ])

        # ============================================================
        # RGB: Visual Stream
        # ============================================================
        self.lh_rgb = VisualEncoder(image_size=rgb_image_size,
            patch_size=rgb_patch_size,
            in_channels=rgb_in_channels,
            embed_dim=rgb_embed_dim,
            depth=depth,
            num_heads=rgb_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.rh_rgb = VisualEncoder(image_size=rgb_image_size,
            patch_size=rgb_patch_size,
            in_channels=rgb_in_channels,
            embed_dim=rgb_embed_dim,
            depth=depth,
            num_heads=rgb_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, lh_pose, rh_pose, lh_rgb, rh_rgb):
        for layer in self.layers:
            lh_rgb, rh_rgb = self.lh_rgb(lh_rgb), self.rh_rgb(rh_rgb)

            lh_pose, rh_pose = layer(lh_pose, rh_pose, lh_rgb, rh_rgb)

        return lh_pose, rh_pose


    """
    Transformer-based encoder for pose sequences.

    Input:
        x: (B, T, input_dim)

    Output:
        x: (B, T, d_model)
    """
        # attn_expand_ratio=2,
        # ffn_expand_ratio=4,
        # conv_expand_ratio=2,
    def __init__(self, input_dim: int, embed_ratio: int, num_heads: int, num_layers: int, mlp_ratio: int, dropout: float, act:str, max_seq_len: int):
        super().__init__()

        d_model = int(embed_ratio * input_dim)

        # Project pose features to Transformer dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer Encoder
        self.transformer = HandEncoder(d_model=d_model, num_heads=num_heads,
            depth=num_layers, 
            mlp_ratio=mlp_ratio, 
            dropout=dropout,
            act=act
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
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
            raise ValueError(    f"Sequence length {T} exceeds "
                f"max_seq_len {self.pos_embedding.size(1)}"
            )

        # (B, T, input_dim)
        #        ↓
        # (B, T, d_model)
        x = self.input_proj(x)

        # Add positional information
        x = x + self.pos_embedding[:, :T, :]

        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)

        # Final normalization
        x = self.norm(x)

        return x