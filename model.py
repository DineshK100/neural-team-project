"""Baseline models for tire-degradation lap-time-loss prediction."""

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """Stacked LSTM with dropout and a linear regression head.

    Input  : (batch, window, feature_dim)
    Output : (batch,) predicted lap-time loss in seconds
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last)).squeeze(-1)


class LinearBaseline(nn.Module):
    """Plain linear regression on the current lap features (no temporal context)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is shaped (B, T, F); use the most recent lap as the non-sequential input.
        return self.fc(x[:, -1, :]).squeeze(-1)
