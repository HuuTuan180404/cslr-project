import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, pe_type, d_model, max_len):
        super().__init__()

        self.pe_type = pe_type

        if pe_type == "sinusoidal":
            self.pe = SinusoidalPE(d_model=d_model, max_len=max_len)

        elif pe_type == "learned":
            self.pe = LearnedTemporalPE(d_model=d_model, max_len=max_len)

        elif pe_type == "continuous":
            self.pe = ContinuousTimePE(d_model=d_model)

        else:
            raise ValueError(f"Unknown PE type: {pe_type}")

    def forward(self, x, timestamps=None):
        if self.pe_type == "continuous":
            if timestamps is None:
                raise ValueError("timestamps is required for continuous PE.")

            return self.pe(x, timestamps)

        return self.pe(x)


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Shape: (max_seq_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Shape: (max_seq_len, 1)
        position = torch.arange(max_len).unsqueeze(1)

        # Shape: (d_model // 2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # Even dimensions: sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # Odd dimensions: cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x + self.pe[:, : x.size(1), :]

        return self.dropout(x)


class LearnedTemporalPE(nn.Module):
    def __init__(self, d_model, max_frames):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn(1, max_frames, d_model) * 0.02)

    def forward(self, x):
        """
        x: [B, L, D]
        """
        L = x.size(1)

        if L > self.pos_embed.size(1):
            raise ValueError(
                f"Sequence length {L} exceeds max_len={self.pos_embed.size(1)}"
            )

        return x + self.pos_embed[:, :L, :]


class ContinuousTimePE(nn.Module):
    def __init__(self, d_model, min_period=0.01, max_period=100.0):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError("d_model must be even.")

        half_dim = d_model // 2

        # Frequencies
        periods = torch.exp(
            torch.linspace(math.log(min_period), math.log(max_period), half_dim)
        )

        frequencies = 2.0 * math.pi / periods

        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, x, timestamps):
        """
        x:
            [B, L, D]

        timestamps:
            [B, L]
            Time of each frame in seconds.

        return:
            [B, L, D]
        """

        B, L, D = x.shape

        if timestamps.shape != (B, L):
            raise ValueError(
                f"Expected timestamps shape [{B}, {L}], got {timestamps.shape}"
            )

        # [B, L, 1]
        t = timestamps.unsqueeze(-1)

        # [B, L, D/2]
        phase = t * self.frequencies

        pe = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

        return x + pe
