from __future__ import annotations

import torch
from torch import nn


class EmotionMLP(nn.Module):
    """Flattened-window MLP baseline for Phase 3 emotion classification."""

    def __init__(
        self,
        sequence_length: int = 30,
        input_features: int = 1434,
        hidden_1: int = 512,
        hidden_2: int = 128,
        num_classes: int = 6,
        dropout_1: float = 0.3,
        dropout_2: float = 0.2,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.input_features = input_features
        self.num_classes = num_classes
        flattened = sequence_length * input_features
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout_1),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout_2),
            nn.Linear(hidden_2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)