import torch
import torch.nn as nn
from config import STATE_DIM, ACTION_DIM

class WorldModel(nn.Module):
    def __init__(self, input_dim=STATE_DIM + ACTION_DIM, hidden_dim=128, output_dim=STATE_DIM + 1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)