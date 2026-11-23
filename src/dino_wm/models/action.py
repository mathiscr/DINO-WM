import torch
from torch import nn
from torch.nn import functional as F

class ActionFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_actions, config.action_dim)
        self.fc2 = nn.Linear(config.action_dim, config.action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
