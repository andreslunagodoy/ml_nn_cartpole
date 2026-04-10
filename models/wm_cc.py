import torch
import torch.nn as nn

class WorldModelCC(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.state_head = nn.Linear(hidden_dim, 4)

        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        delta = self.state_head(features)
        reward = self.reward_head(features)
        return delta, reward
