import numpy as np
import torch

from ray.rllib.algorithms.sac import SAC


class LowLevelController:

    def __init__(self, agents):

        self.agents = agents

        # LOAD TRAINED SAC-LSTM CHECKPOINT
        self.algo = SAC.from_checkpoint(
            "E:/uav_hmarl_project/sac_checkpoint_final"
        )

        # hidden states for each UAV
        self.hidden_states = {
            agent: [
                np.zeros(256, np.float32),
                np.zeros(256, np.float32)
            ]
            for agent in agents
        }

    def act(self, obs, intent, hidden, macro_action):

        x = np.concatenate([
            obs,
            intent,
            macro_action
        ]).astype(np.float32)

        x = torch.FloatTensor(x).unsqueeze(0)

        with torch.no_grad():

            action, hidden = self.model(x, hidden)

        return action.squeeze(0).numpy(), hidden

  