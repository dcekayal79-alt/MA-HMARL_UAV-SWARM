import ray
import matplotlib.pyplot as plt
import numpy as np

from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env

from rsacenv import UAVRSACEnv

# =========================================
# INIT RAY
# =========================================
ray.init(ignore_reinit_error=True)

# =========================================
# REGISTER ENV
# =========================================
def env_creator(config):
    return UAVRSACEnv()

register_env("low_level_env", env_creator)

# =========================================
# CONFIG
# =========================================
config = (
    SACConfig()

    .environment("low_level_env")

    .framework("torch")

    .env_runners(
        num_env_runners=1
    )

    .training(
        gamma=0.99,
        train_batch_size=256,

        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,

        replay_buffer_config={
            "capacity": 100000
        },

        tau=0.005,
    )

    .resources(num_gpus=0)
)

# =========================================
# BUILD
# =========================================
algo = config.build_algo()

# =========================================
# TRAIN
# =========================================
rewards = []

for i in range(20):

    result = algo.train()

    reward = result["env_runners"]["episode_return_mean"]

    rewards.append(reward)

    print(f"Iter {i}: reward = {reward:.2f}")

    if i % 10 == 0:

        checkpoint = algo.save(
            "E:/uav_hmarl_project/sac_checkpoint"
        )

       

# =========================================
# FINAL SAVE
# =========================================
checkpoint = algo.save(
    "E:/uav_hmarl_project/sac_checkpoint_final"
)



# =========================================
# PLOT 1
# =========================================
plt.figure(figsize=(8,5))

plt.plot(rewards)

plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.title("RSAC Training Curve")

plt.grid()

plt.show()

# =========================================
# PLOT 2 (SMOOTHED)
# =========================================
window = 5

if len(rewards) >= window:

    smoothed = np.convolve(
        rewards,
        np.ones(window)/window,
        mode='valid'
    )

    plt.figure(figsize=(8,5))

    plt.plot(smoothed)

    plt.xlabel("Iteration")
    plt.ylabel("Smoothed Reward")

    plt.title("Smoothed RSAC Reward Curve")

    plt.grid()

    plt.show()

# =========================================
# SHUTDOWN
# =========================================
ray.shutdown()