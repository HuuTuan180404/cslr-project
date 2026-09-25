import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba


class MambaTextEncoder(nn.Module):
    """
    Mamba-based text encoder for visual-textual contrastive learning.

    Input:
        input_ids: [B, L]
        attention_mask: [B, L]

    Output:
        token_features: [B, L, D]
        sentence_features: [B, D]
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        mamba_layers,
        state_dim,
        conv_kernel,
        expand,
        dropout,
        padding_idx,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx

        # --------------------------------------------------
        # 1. Token Embedding
        # --------------------------------------------------
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        # --------------------------------------------------
        # 2. Mamba blocks
        # --------------------------------------------------
        self.mamba_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=embed_dim,
                    d_state=state_dim,
                    d_conv=conv_kernel,
                    expand=expand,
                )
                for _ in range(mamba_layers)
            ]
        )

        # LayerNorm after each Mamba block
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(mamba_layers)]
        )

        self.dropout = nn.Dropout(dropout)

        # --------------------------------------------------
        # 3. Final normalization
        # --------------------------------------------------
        self.final_norm = nn.LayerNorm(embed_dim)

    def masked_mean_pooling(self, x, attention_mask) -> torch.Tensor:
        """
        x:
            [B, L, D]

        attention_mask:
            [B, L]
            1 = valid token
            0 = padding
        """

        mask = attention_mask.unsqueeze(-1).float()
        # [B, L, 1]

        x = x * mask

        summed = x.sum(dim=1)
        # [B, D]

        count = mask.sum(dim=1).clamp(min=1e-6)
        # [B, 1]

        return summed / count

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids:
                [B, L]

            attention_mask:
                [B, L]

        Returns:
            token_features:
                [B, L, D]

            sentence_features:
                [B, D]
        """

        # --------------------------------------------------
        # Token embedding
        # --------------------------------------------------
        x = self.embedding(input_ids)
        # [B, L, D]

        x = self.embedding_dropout(x)

        # --------------------------------------------------
        # Zero-out padding tokens
        # --------------------------------------------------
        mask = attention_mask.unsqueeze(-1).float()
        x = x * mask  # [B, L, D]

        # --------------------------------------------------
        # Mamba blocks
        # --------------------------------------------------
        for mamba, norm in zip(self.mamba_layers, self.norm_layers):
            residual = x

            x = mamba(x)
            # [B, L, D]

            x = self.dropout(x)

            x = residual + x

            x = norm(x)

            # Keep padding positions zero
            x = x * mask

        # --------------------------------------------------
        # Final normalization
        # --------------------------------------------------
        x = self.final_norm(x)

        # --------------------------------------------------
        # Sentence-level representation
        # --------------------------------------------------
        sentence_features = self.masked_mean_pooling(
            x,
            attention_mask,
        )
        # [B, D]

        return x, sentence_features


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        x: [B, D]
        """

        x = self.projection(x)
        # [B, E]

        x = F.normalize(x, dim=-1)

        return x


class TextBranch(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        mamba_layers,
        projection_dim,
        state_dim,
        conv_kernel,
        expand,
        dropout,
        padding_idx,
    ):
        super().__init__()

        self.encoder = MambaTextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            mamba_layers=mamba_layers,
            padding_idx=padding_idx,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            expand=expand,
            dropout=dropout,
        )

        self.projection = ProjectionHead(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=projection_dim,
            dropout=dropout,
        )

    def forward(self, input_ids, attention_mask):

        token_features, sentence_features = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        z_text = self.projection(sentence_features)

        return {
            "token_features": token_features,
            "sentence_features": sentence_features,
            "z_text": z_text,
        }
