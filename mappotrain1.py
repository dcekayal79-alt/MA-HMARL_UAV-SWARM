from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from env5 import UAVHMARLEnv
import ray
import matplotlib.pyplot as plt
import numpy as np

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
# MULTI-AGENT SETUP
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
# CONFIG (STABLE VERSION)
# =========================
config = (
    PPOConfig()
    .environment("uav_hmarl")
    .framework("torch")

    .env_runners(
        num_env_runners=1,
        rollout_fragment_length=200,
    )

       .training(
        lr=5e-5,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,

        entropy_coeff=0.01,
        vf_loss_coeff=0.5,

        # ✅ NEW RLlib API minibatch settings
        train_batch_size=1000,
        minibatch_size=256,
        num_epochs=10,
    )

    .reporting(
        min_sample_timesteps_per_iteration=1000,
    )

    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
    )
)

# =========================
# BUILD ALGO
# =========================
algo = config.build()

# =========================
# TRAIN LOOP
# =========================
rewards_history = []

for i in range(50):
    res = algo.train()

    reward = (
        res.get("env_runners", {})
           .get("episode_return_mean",
                res.get("episode_return_mean",
                        res.get("episode_reward_mean", None)))
    )

    rewards_history.append(reward)

    print(f"Iter {i}: Reward = {reward}")

# =========================
# PLOT RAW CURVE
# =========================
plt.figure()
plt.plot(rewards_history)
plt.xlabel("Iteration")
plt.ylabel("Mean Episode Reward")
plt.title("Training Reward Curve")
plt.grid()
plt.show()

# =========================
# SMOOTHED CURVE
# =========================
def moving_average(data, window=5):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')

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