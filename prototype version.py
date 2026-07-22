import gymnasium as gym
from gymnasium import spaces
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt

# Define the UAV Swarm Environment
class UAVSwarmEnv(gym.Env):
    def __init__(self, num_uavs=10, state_dim=8, action_dim=5):
        super(UAVSwarmEnv, self).__init__()

        self.num_uavs = num_uavs
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_uavs, self.state_dim), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_uavs, self.action_dim), dtype=np.float32)

        self.state = np.random.randn(self.num_uavs, self.state_dim)
        self.trajectories = [[] for _ in range(self.num_uavs)]  # Store UAV trajectories

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.randn(self.num_uavs, self.state_dim)
        self.trajectories = [[self.state[i, :2].copy()] for i in range(self.num_uavs)]  # Reset trajectories
        return self.state, {}  # Ensure it returns (observation, info) tuple

    def step(self, actions):
         actions = np.asarray(actions)

         # Reshape if SB3 passes a flattened action vector
         if actions.ndim == 1:
            if actions.size == self.action_dim:
              # Repeat the same action for every UAV
              actions = np.tile(actions, (self.num_uavs, 1))
            else:
              actions = actions.reshape(self.num_uavs, self.action_dim)

         next_state = self.state.copy()
         next_state[:, :self.action_dim] += actions * 0.1
         next_state += np.random.randn(*self.state.shape) * 0.01

         goal = np.array([10, 10])
         distance_to_goal = np.linalg.norm(next_state[:, :2] - goal, axis=1)

         proximity_reward = np.maximum(0, 10 - distance_to_goal.mean()) * 2
         action_penalty = -0.02 * np.linalg.norm(actions, axis=1).mean()
         time_penalty = -0.01
         goal_reached_bonus = 10 if distance_to_goal.mean() < 1.0 else 0

         reward = (
             proximity_reward
             + action_penalty
             + time_penalty
             + goal_reached_bonus
         ) / 5

         terminated = distance_to_goal.mean() < 0.5
         truncated = False

         self.state = next_state

         for i in range(self.num_uavs):
             self.trajectories[i].append(self.state[i, :2].copy())

         return next_state, reward, terminated, truncated, {}

    def render(self):
        plt.figure(figsize=(8, 8))
        for traj in self.trajectories:
            traj = np.array(traj)
            plt.plot(traj[:, 0], traj[:, 1], marker="o", markersize=2, alpha=0.7)

        plt.scatter([10], [10], color="red", marker="*", s=200, label="Goal")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("UAV Trajectories")
        plt.legend()
        plt.grid()
        plt.show()

# Define the Hybrid RL Agent
class HybridRL:
    def __init__(self):
        print("Using cpu device")
        self.env = make_vec_env(UAVSwarmEnv, n_envs=1)  # Single env for better visualization

        # MAPPO for high-level decision-making
        self.mappo = PPO('MlpPolicy', self.env, verbose=1, device="cpu")

        # SAC for low-level UAV control
        self.sac = SAC('MlpPolicy', self.env, verbose=1, device="cpu", learning_starts=10000, train_freq=(1000, "step"))

        self.rewards = []  # Store rewards per episode

    def train(self, episodes=50):
        for episode in range(episodes):
            state = self.env.reset()[0]  # FIXED: Extracts the observation
            total_reward = 0
            done = np.array([False] * self.env.num_envs)
            step_count = 0
            max_steps = 100

            while not done.any() and step_count < max_steps:
                high_level_action, _ = self.mappo.predict(state, deterministic=True)
                refined_action, _ = self.sac.predict(state, deterministic=True)

                next_state, reward, done, info = self.env.step(refined_action)

                total_reward += reward.mean()
                step_count += 1

                state = next_state  # Ensure state updates
                done = done.any()  # Convert `done` to boolean

            self.rewards.append(total_reward)
            print(f"Train Episode {episode + 1}, Total Reward: {total_reward}")

    def test(self, episodes=10):
        for episode in range(episodes):
            state = self.env.reset()[0]  # FIXED: Extracts the observation
            total_reward = 0
            done = np.array([False] * self.env.num_envs)
            step_count = 0
            max_steps = 100

            while not done.any() and step_count < max_steps:
                high_level_action, _ = self.mappo.predict(state, deterministic=True)
                refined_action, _ = self.sac.predict(state, deterministic=True)

                next_state, reward, done, info = self.env.step(refined_action)

                total_reward += reward.mean()
                step_count += 1

                state = next_state  # Ensure state updates
                done = done.any()  # Convert `done` to boolean

                if step_count >= max_steps:
                    break

            self.rewards.append(total_reward)
            print(f"Test Episode {episode + 1}, Total Reward: {total_reward}")

    def plot_rewards(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.rewards, label='Total Reward per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Reward Curve for MAPPO + SAC UAV Swarm')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_trajectories(self):
        self.env.envs[0].render()  # Call render function in first environment

# Run the Hybrid RL Training & Testing
if __name__ == "__main__":
    hybrid_rl = HybridRL()
    hybrid_rl.train(episodes=50)
    hybrid_rl.test(episodes=10)
    hybrid_rl.plot_rewards()  # Show reward curve
    hybrid_rl.plot_trajectories()  # Show UAV movement trajectories
