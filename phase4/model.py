from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_length: int = 512) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")

        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embedding_dim))
        encoding = torch.zeros(max_length, embedding_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1)]


class EmotionTransformer(nn.Module):
    """Temporal transformer baseline for Phase 4 emotion classification."""

    def __init__(
        self,
        sequence_length: int = 30,
        input_features: int = 1434,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        num_classes: int = 6,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if input_features <= 0:
            raise ValueError(f"input_features must be positive, got {input_features}")
        if model_dim <= 0:
            raise ValueError(f"model_dim must be positive, got {model_dim}")
        if num_heads <= 0 or model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if feedforward_dim <= 0:
            raise ValueError(f"feedforward_dim must be positive, got {feedforward_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        self.sequence_length = sequence_length
        self.input_features = input_features
        self.model_dim = model_dim
        self.num_classes = num_classes

        self.input_projection = nn.Linear(input_features, model_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(model_dim, max_length=max(max_length, sequence_length))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.classifier(x)
