import ray
import matplotlib.pyplot as plt
import numpy as np

from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env

from rsacenv import UAVRSACEnv


# =========================================
# REGISTER ENV
# =========================================
def env_creator(config):
    return UAVRSACEnv()

register_env(
    "low_level_env",
    env_creator
)

# =========================================
# CONFIG
# =========================================
config = (

    SACConfig()
    .api_stack(
    05:31 PM 09-05-2026enable_rl_module_and_learner=False,
    enable_env_runner_and_connector_v2=False
    )

    .environment(
        env="low_level_env"
    )

    .framework("torch")

    # =====================================
    # ENV RUNNERS
    # =====================================
    .env_runners(
        num_env_runners=1
    )

    # =====================================
    # TRAINING
    # =====================================
    .training(

        # -----------------------------
        # RL SETTINGS
        # -----------------------------
        gamma=0.99,

        train_batch_size=256,

        actor_lr=3e-4,

        critic_lr=3e-4,

        alpha_lr=3e-4,

        tau=0.005,

        # -----------------------------
        # REPLAY BUFFER
        # -----------------------------
        replay_buffer_config={
            "capacity": 100000
        },

        # =================================
        # RECURRENT SAC (RSAC)
        # =================================
        model={

            # recurrent memory
            "use_lstm": True,

            "lstm_cell_size": 128,

            "max_seq_len": 20,

            # improves temporal inference
            "lstm_use_prev_action": True,

            "lstm_use_prev_reward": True,

            # feature extractor
            "fcnet_hiddens": [256, 256],

            "fcnet_activation": "tanh",
        },
    )

    .resources(
        num_gpus=0
    )
)

# =========================================
# BUILD ALGORITHM
# =========================================
algo = config.build_algo()

# =========================================
# TRAIN LOOP
# =========================================
rewards = []

num_iterations = 200

for i in range(num_iterations):

    result = algo.train()

    reward = result["env_runners"][
        "episode_return_mean"
    ]

    rewards.append(reward)

    print(
        f"Iter {i}: reward = {reward:.2f}"
    )

    # =====================================
    # CHECKPOINT
    # =====================================
    if i % 50 == 0:

        algo.save(
            "E:/uav_hmarl_project/sac_checkpoint"
        )

# =========================================
# FINAL SAVE
# =========================================
algo.save(
    "E:/uav_hmarl_project/sac_checkpoint_final"
)

# =========================================
# RAW REWARD PLOT
# =========================================
plt.figure(figsize=(8, 5))

plt.plot(rewards)

plt.xlabel("Training Iteration")

plt.ylabel("Reward")

plt.title(
    "RSAC Training Curve"
)

plt.grid()

plt.savefig(
    "E:/uav_hmarl_project/rsac_reward_curve.png"
)

plt.show()

# =========================================
# SMOOTHED PLOT
# =========================================
window = 10

if len(rewards) >= window:

    smoothed = np.convolve(
        rewards,
        np.ones(window) / window,
        mode="valid"
    )

    plt.figure(figsize=(8, 5))

    plt.plot(smoothed)

    plt.xlabel("Training Iteration")

    plt.ylabel("Smoothed Reward")

    plt.title(
        "Smoothed RSAC Reward Curve"
    )

    plt.grid()

    plt.savefig(
        "E:/uav_hmarl_project/rsac_smoothed_curve.png"
    )

    plt.show()

# =========================================
# SHUTDOWN
# =========================================
ray.shutdown()