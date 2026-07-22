from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from env4 import UAVHMARLEnv
import ray

# =========================
# INIT RAY
# =========================
ray.init(ignore_reinit_error=True)

# =========================
# ENV REGISTRATION
# =========================
def env_creator(config):
    return ParallelPettingZooEnv(UAVHMARLEnv(num_uavs=5))

register_env("uav_hmarl", env_creator)

# =========================
# MULTI-AGENT SETUP (DEFINE OUTSIDE)
# =========================
policies = {
    "leader_policy": (None, None, None, {}),
    "worker_policy": (None, None, None, {}),
}

def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == "uav_0":
        return "leader_policy"
    else:
        return "worker_policy"

# =========================
# CONFIG (FIXED)
# =========================
config = (
    PPOConfig()
    .environment("uav_hmarl")
    .framework("torch")

    .env_runners(
        num_env_runners=1,
        rollout_fragment_length=200
    )

    .training(
        train_batch_size=1000,
        gamma=0.99,
        lr=1e-4,
        entropy_coeff=0.01 
    )

    .reporting(
        min_sample_timesteps_per_iteration=1000
    )

    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
    )
)

# =========================
# BUILD ALGO
# =========================
algo = config.build_algo()

# =========================
# TRAIN LOOP
# =========================
import matplotlib.pyplot as plt

rewards_history = []

for i in range(20):
    res = algo.train()

    reward = (
        res.get("env_runners", {})
           .get("episode_return_mean",
                res.get("episode_return_mean",
                        res.get("episode_reward_mean", None)))
    )

    rewards_history.append(reward)

    print(f"Iter {i}: Reward = {reward}")



plt.figure()
plt.plot(rewards_history)
plt.xlabel("Iteration")
plt.ylabel("Mean Episode Reward")
plt.title("Training Reward Curve")
plt.grid()
plt.show()


import numpy as np

def moving_average(data, window=5):
    return np.convolve(data, np.ones(window)/window, mode='valid')

smoothed = moving_average(rewards_history, window=5)

plt.figure()
plt.plot(rewards_history, alpha=0.3, label="Raw")
plt.plot(range(len(smoothed)), smoothed, label="Smoothed", linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.title("Smoothed Training Curve")
plt.legend()
plt.grid()
plt.show()


# =========================
# SAVE MODEL
# =========================
algo.save("E:/uav_hmarl_project/macro_policy")

# =========================
# SHUTDOWN
# =========================
ray.shutdown()