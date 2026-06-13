"""TabularMLP — original EAGF tabular classifier."""
import torch.nn as nn


class TabularMLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64), n_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], hidden[2]),
            nn.ReLU(),
            nn.Linear(hidden[2], n_classes),
        )

    def forward(self, x):
        return self.net(x)
