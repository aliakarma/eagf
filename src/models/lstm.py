"""Bidirectional LSTM classifier for CS2 IIoT intrusion detection.

Paper Section 5.2, Table 12: 2-layer bidirectional LSTM, hidden_size=128,
Dense + LayerNorm head.
"""
import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """2-layer bidirectional LSTM with Dense + LayerNorm head.

    Architecture (paper Section 5.2, Table 12):
        - 2-layer bidirectional LSTM, hidden_size=128
        - Head: Linear(hidden*2, hidden) → LayerNorm → ReLU → Dropout → Linear(hidden, n_classes)

    If input is 2D (batch, features), it is treated as a single-timestep
    sequence (batch, 1, features) for LSTM processing.
    """

    def __init__(self, input_dim: int, hidden_size: int = 128,
                 num_layers: int = 2, n_classes: int = 2,
                 dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        direction_factor = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        head_input = hidden_size * direction_factor
        self.head = nn.Sequential(
            nn.Linear(head_input, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, seq_len, features) or (batch, features).
               If 2D, unsqueezed to (batch, 1, features).
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.head(last_hidden)
