import ray
import matplotlib.pyplot as plt

from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

from partenv import UAVHMARLEnv


# ==========================================
# RAY INIT
# ==========================================
ray.init(ignore_reinit_error=True)

rewards_history = []


# ==========================================
# ENV CREATOR
# ==========================================
def env_creator(config):

    env = UAVHMARLEnv(
        num_uavs=5,
        dropout_prob=0.7
    )

    return ParallelPettingZooEnv(env)


register_env("uav_hmarl", env_creator)

temp_env = env_creator({})

obs_space = temp_env.observation_space["uav_0"]
act_space = temp_env.action_space["uav_0"]


# ==========================================
# POLICY MAPPING
# ==========================================
def policy_mapping_fn(agent_id, *args, **kwargs):

    if agent_id == "uav_0":
        return "leader_policy"

    return "worker_policy"


# ==========================================
# POLICIES
# ==========================================
policies = {

    "leader_policy": (
        None,
        obs_space,
        act_space,
        {}
    ),

    "worker_policy": (
        None,
        obs_space,
        act_space,
        {}
    ),
}


# ==========================================
# PPO CONFIG
# ==========================================
config = (

    PPOConfig()

    # ======================================
    # OLD STABLE RLlib API
    # ======================================
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )

    .environment("uav_hmarl")

    .framework("torch")

    # ======================================
    # MULTI AGENT
    # ======================================
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
    )

    # ======================================
    # TRAINING
    # ======================================
    .training(

        gamma=0.99,

        lr=1e-5,

        train_batch_size=4000,

        num_epochs=10,

        grad_clip=0.5,

        model={

            # ==============================
            # ENABLE LSTM
            # ==============================
            "use_lstm": True,

            "lstm_cell_size": 256,

            "max_seq_len": 40,

            "lstm_use_prev_action": True,

            "lstm_use_prev_reward": True,

            # ==============================
            # FC NETWORK
            # ==============================
            "fcnet_hiddens": [256, 256],

            "fcnet_activation": "relu",
        }
    )

    # ======================================
    # ROLLOUTS
    # ======================================
    .env_runners(

        num_env_runners=1,

        rollout_fragment_length=40,

        batch_mode="truncate_episodes",
    )

    .resources(num_gpus=0)
)


# ==========================================
# BUILD ALGORITHM
# ==========================================
algo = config.build()


# ==========================================
# TRAIN LOOP
# ==========================================
for i in range(10):

    result = algo.train()

    reward = result["env_runners"]["episode_reward_mean"]

    rewards_history.append(reward)

    print(f"Iter {i}: reward = {reward:.2f}")

    # ======================================
    # SAVE CHECKPOINT
    # ======================================
    if i % 10 == 0:

        checkpoint = algo.save(
            "E:/uav_hmarl_project/mappo_lstm_checkpoint"
        )

        print(
            "Saved checkpoint at:",
            checkpoint.checkpoint.path
        )


# ==========================================
# FINAL CHECKPOINT
# ==========================================
final_checkpoint = algo.save(
    "E:/uav_hmarl_project/mappo_lstm_final"
)

print(
    "Final checkpoint:",
    final_checkpoint.checkpoint.path
)


# ==========================================
# PLOT REWARD CURVE
# ==========================================
plt.figure(figsize=(10, 6))

plt.plot(
    rewards_history,
    linewidth=2,
    label="MAPPO + LSTM"
)

plt.xlabel("Training Iteration")

plt.ylabel("Mean Reward")

plt.title(
    "MAPPO + LSTM under Partial Observability"
)

plt.grid(True)

plt.legend()

plt.savefig(
    "E:/uav_hmarl_project/reward_curve.png"
)

plt.show()


# ==========================================
# SHUTDOWN
# ==========================================
ray.shutdown()