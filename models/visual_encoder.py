import torch
import torch.nn as nn
from typing import Tuple

from .utils import MLP
from .transformer import TransformerEncoderLayer


class PatchEmbedding(nn.Module):
    """
    Image -> Patch tokens

    Input:
        x: (B, 3, H, W)

    Output:
        x: (B, N, D)
        N = number of patches
        D = embedding dimension
    """

    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patches = (image_size // patch_size) ** 2

        # Conv2d is used only as an efficient way to perform
        # patch extraction + linear projection.
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) # (B, 3, H, W)
        x = x.flatten(2) # (B, D, H/P, W/P)
        x = x.transpose(1, 2) # (B, D, N) -> (B, N, D)
        return x


class ViT(nn.Module):
    """
    Vision Transformer implemented from scratch.

    Input:
        x: (B, 3, 224, 224)

    Output:
        logits: (B, num_classes)
    """

    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3, num_classes: int = 1000, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(image_size=image_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        # 2. CLS Token: (1, 1, D)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. Positional Embedding [+1 because of CLS token]: (1, N+1, D)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_dropout = nn.Dropout(dropout)

        # 4. Transformer Encoder
        self.blocks = nn.ModuleList([
            VisualTransformerEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # 1. Image -> Patch Embeddings: (B, 3, 112, 112) -> (B, 49, 768)
        x = self.patch_embed(x)

        # 2. Add CLS token: (1, 1, D) -> (B, 1, D)
        cls_token = self.cls_token.expand(B, -1, -1)

        # Cat: (B, 1, D) + (B, 49, D) -> (B, 50, D)
        x = torch.cat((cls_token, x), dim=1)

        # 3. Add positional embedding (B, 50, D)
        x = x + self.pos_embed

        x = self.pos_dropout(x) # (B, 50, D)

        # 4. Transformer Encoder
        for block in self.blocks:
            x = block(x, pre_norm=True)

        # 5. Final LayerNorm
        x = self.norm(x)

        # 6. Take CLS token (B, 50, D) -> (B, D)
        cls = x[:, 0]

        # 7. Classification: (B, D) -> (B, num_classes)
        logits = self.head(cls)

        return logits


class VisualEncoder(nn.Module):
    """
    Vision Transformer implemented from scratch.

    Input:
        x: (B, 3, 224, 224)

    Output:
        logits: (B, num_classes)
    """

    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3, act: str = "gelu", embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.0, pre_norm: bool = False):
        super().__init__()

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(image_size=image_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        # 2. CLS Token: (1, 1, D)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. Positional Embedding [+1 because of CLS token]: (1, N+1, D)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_dropout = nn.Dropout(dropout)

        # 4. Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embed_dim, num_heads=num_heads, act=act, mlp_ratio=mlp_ratio, dropout=dropout, pre_norm=pre_norm)
            for _ in range(depth)
        ])

        self.count_layer = 0

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, pre_norm: bool) -> torch.Tensor:
        if self.count_layer == len(self.blocks):
            raise ValueError(
                f"count_layer ({self.count_layer}) must be smaller than "
                f"the number of blocks ({len(self.blocks)})"
            )

        if self.count_layer == 0:
            B = x.shape[0]

            # 1. Image -> Patch Embeddings: (B, 3, 112, 112) -> (B, 49, 768)
            x = self.patch_embed(x)

            # 2. Add CLS token: (1, 1, D) -> (B, 1, D)
            cls_token = self.cls_token.expand(B, -1, -1)

            # Cat: (B, 1, D) + (B, 49, D) -> (B, 50, D)
            x = torch.cat((cls_token, x), dim=1)

            # 3. Add positional embedding (B, 50, D)
            x = x + self.pos_embed

            x = self.pos_dropout(x) # (B, 50, D)

        x, _ = self.blocks[self.count_layer](x)
        self.count_layer += 1

        return x


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":

    model = ViT(image_size=112, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1)

    x = torch.randn(2, 3, 112, 112)

    logits = model(x)

    print("Input :", x.shape)
    print("Output:", logits.shape)

    # Expected:
    # Input : torch.Size([2, 3, 112, 112])
    # Output: torch.Size([2, 1000])
