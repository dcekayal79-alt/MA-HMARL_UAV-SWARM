# low_level_rsac.py

import numpy as np
import torch
import torch.nn as nn


class RSACLite(nn.Module):
    def __init__(self, obs_dim=12, intent_dim=5, hidden_dim=64, action_dim=3):
        super().__init__()

        self.input_dim = obs_dim + intent_dim

        self.rnn = nn.GRU(self.input_dim, hidden_dim, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        action = self.fc(out[:, -1, :])
        return action, h


class LowLevelController:
    def __init__(self, agents, obs_dim=12, intent_dim=5):
        self.device = torch.device("cpu")

        self.model = RSACLite(
            obs_dim=obs_dim,
            intent_dim=intent_dim
        ).to(self.device)

        # Hidden state per agent
        self.hidden_states = {
            a: torch.zeros(1, 1, 64).to(self.device)
            for a in agents
        }

    def reset(self):
        for a in self.hidden_states:
            self.hidden_states[a] = torch.zeros(1, 1, 64).to(self.device)

    def process_intent(self, intent):
        """
        Normalize distance for stability
        intent = [dx, dy, dz, distance, reached]
        """

        intent = intent.copy()

        # Normalize distance (important)
        intent[3] = np.tanh(intent[3] / 10.0)

        return intent

    def act(self, agent, obs, intent):
        """
        obs: (12,)
        intent: (5,)
        """

        intent = self.process_intent(intent)

        x = np.concatenate([obs, intent])  # (17,)
        x = torch.tensor(x, dtype=torch.float32).view(1, 1, -1).to(self.device)

        h = self.hidden_states[agent]

        with torch.no_grad():
            action, new_h = self.model(x, h)

        self.hidden_states[agent] = new_h

        action = action.squeeze().cpu().numpy()

        # small exploration noise
        action += np.random.normal(0, 0.03, size=3)

        return np.clip(action, -1, 1)

if __name__ == "__main__":

    agents = ["uav_0", "uav_1"]

    controller = LowLevelController(agents)

    # Dummy observation (12D)
    obs = {
        "uav_0": np.random.rand(12),
        "uav_1": np.random.rand(12),
    }

    # Dummy intent (5D) ← matches your new design
    intent = {
        "uav_0": np.array([0.7, 0.7, 0, 5.0, 0]),
        "uav_1": np.array([0.5, -0.5, 0, 3.0, 0]),
    }

    for step in range(5):
        print(f"\nStep {step}")

        for a in agents:
            action = controller.act(a, obs[a], intent[a])
            print(f"{a} action:", action)