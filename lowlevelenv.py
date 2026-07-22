import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box


class LowLevelEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()

        # Example dimensions (adjust if needed)
        self.obs_dim = 8   # e.g. position (3) + velocity (3) + intent (2)
        self.act_dim = 3   # UAV control (vx, vy, vz)

        # ✅ FIX: proper dtype
        self.observation_space = Box(
            low=-10.0,
            high=10.0,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.act_dim,),
            dtype=np.float32
        )

        self.state = None
        self.step_count = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.random.uniform(-1, 1, size=self.obs_dim)

        self.step_count = 0

        # ✅ FIX: float32
        return self.state.astype(np.float32), {}

    def step(self, action):
        self.step_count += 1

        # Simple dynamics (replace with UAV logic)
        self.state[:3] += action  # move position

        # Reward: move toward origin
        reward = -np.linalg.norm(self.state[:3])

        terminated = self.step_count >= self.max_steps
        truncated = False

        # ✅ FIX: float32
        return self.state.astype(np.float32), reward, terminated, truncated, {}