import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mean = nn.Linear(128, act_dim)
        self.log_std = nn.Linear(128, act_dim)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -20, 2)
        return mean, log_std


class LowLevelController:

    def __init__(self, obs_dim=12, act_dim=3):
        self.actor = Actor(obs_dim + act_dim, act_dim)

    def act(self, obs, intent):
        x = np.concatenate([obs, intent])
        x = torch.FloatTensor(x).unsqueeze(0)

        mean, log_std = self.actor(x)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()

        return torch.tanh(action).detach().numpy()[0]