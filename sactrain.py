import ray
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env

from lowlevelenv import LowLevelEnv

# =========================
# INIT
# =========================
ray.init(ignore_reinit_error=True)

# =========================
# REGISTER ENV
# =========================
register_env("low_level_env", lambda cfg: LowLevelEnv())

# =========================
# CONFIG (NEW RLlib API)
# =========================
config = (
    SACConfig()
    .environment("low_level_env")
    .framework("torch")

    # ✅ FIX: new API
    .env_runners(num_env_runners=1)

    # ✅ SAC-specific learning rates
    .training(
        gamma=0.99,
        train_batch_size=256,

        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
    )

    .resources(num_gpus=0)
)

# ✅ FIX: new build API
algo = config.build_algo()

# =========================
# TRAIN LOOP
# =========================
rewards = []

for i in range(20):
    result = algo.train()

    reward = result["env_runners"]["episode_return_mean"]
    rewards.append(reward)

    print(f"Iter {i}: reward = {reward:.2f}")

    if i % 20 == 0:
        checkpoint = algo.save("E:/uav_hmarl_project/sac_checkpoint")
        print("Saved:", checkpoint)

# final save
checkpoint = algo.save("E:/uav_hmarl_project/sac_checkpoint_final")
print("Final checkpoint:", checkpoint)


import matplotlib.pyplot as plt

plt.plot(rewards)
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.title("SAC Training Curve")
plt.show()


import numpy as np

window = 5
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

plt.plot(smoothed)
plt.xlabel("Iteration")
plt.ylabel("Smoothed Reward")
plt.title("Smoothed SAC Reward Curve")
plt.show()

# =========================
# CLEANUP
# =========================
ray.shutdown()