import os
import random

import numpy as np
import torch
import ray

from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from rsacenv import UAVRSACEnv
from rsac_actor import RecurrentSACActor
from rsac_qmodel import RecurrentSACQModel


# ============================================================
# SETTINGS
# ============================================================

ENV_NAME = "low_level_env"

CHECKPOINT_DIR = (
    "E:/uav_hmarl_project/rsac_recurrent_final"
)

SEED = 42

NUM_ITERATIONS = 100

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)


# ============================================================
# ENVIRONMENT
# ============================================================

def env_creator(config):
    return UAVRSACEnv()


register_env(
    ENV_NAME,
    env_creator
)


# ============================================================
# MODEL REGISTRATION
# ============================================================

ModelCatalog.register_custom_model(
    "recurrent_sac_actor",
    RecurrentSACActor
)

ModelCatalog.register_custom_model(
    "recurrent_sac_q",
    RecurrentSACQModel
)


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

print("\n==============================================")
print("TESTING RSAC ENVIRONMENT")
print("==============================================")

test_env = UAVRSACEnv()

print(
    "Observation space:",
    test_env.observation_space
)

print(
    "Action space:",
    test_env.action_space
)

print(
    "Observation shape:",
    test_env.observation_space.shape
)

print(
    "Action shape:",
    test_env.action_space.shape
)

assert test_env.observation_space.shape == (13,)
assert test_env.action_space.shape == (3,)

obs, info = test_env.reset(seed=SEED)

print(
    "Initial observation shape:",
    obs.shape
)

assert obs.shape == (13,)

print("Environment check PASSED")


# ============================================================
# SAC CONFIGURATION
# ============================================================

print("\n==============================================")
print("BUILDING RECURRENT SAC")
print("==============================================")

config = (
    SACConfig()

    # --------------------------------------------------------
    # IMPORTANT:
    # Use RLlib OLD API stack because we are using
    # TorchModelV2 / RecurrentNetwork.
    # --------------------------------------------------------
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )

    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------
    .environment(
        env=ENV_NAME,
    )

    # --------------------------------------------------------
    # PYTORCH
    # --------------------------------------------------------
    .framework("torch")

    # --------------------------------------------------------
    # ROLLOUT
    # --------------------------------------------------------
    .env_runners(
        num_env_runners=0,
        rollout_fragment_length=100,
    )
    # --------------------------------------------------------
    # SAC SETTINGS
    # --------------------------------------------------------
    .training(

        twin_q=True,

        gamma=0.99,

        train_batch_size=256,

        actor_lr=3e-4,

        critic_lr=3e-4,

        alpha_lr=3e-4,

        tau=0.005,

        target_entropy="auto",

        num_steps_sampled_before_learning_starts=1000,

        replay_buffer_config={
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": 100000,
        },

        # ----------------------------------------------------
        # RECURRENT ACTOR
        # ----------------------------------------------------
        policy_model_config={
            "custom_model": "recurrent_sac_actor",
            "custom_model_config": {},

            # Sequence length for recurrent training.
            "max_seq_len": 20,
        },

        # ----------------------------------------------------
        # RECURRENT Q NETWORK
        # ----------------------------------------------------
        q_model_config={
            "custom_model": "recurrent_sac_q",
            "custom_model_config": {},

            "max_seq_len": 20,
        },
    )

    # --------------------------------------------------------
    # CPU ONLY
    # --------------------------------------------------------
    .resources(
        num_gpus=0,
    )
)


# ============================================================
# BUILD
# ============================================================

print("\n==============================================")
print("BUILDING SAC ALGORITHM")
print("==============================================")

ray.init(
    ignore_reinit_error=True,
    include_dashboard=False,
)

algo = config.build()

print("\n==============================================")
print("RECURRENT SAC BUILT SUCCESSFULLY")
print("==============================================")

print(
    "Algorithm:",
    type(algo).__name__
)

print(
    "Ray version:",
    ray.__version__
)

print(
    "Seed:",
    SEED
)


# ============================================================
# TRAINING
# ============================================================

print("\n==============================================")
print("STARTING RECURRENT SAC TRAINING")
print("==============================================")

for iteration in range(1, NUM_ITERATIONS + 1):

    result = algo.train()

    env_results = result.get(
        "env_runners",
        {}
    )

    episode_reward = env_results.get(
        "episode_reward_mean",
        None
    )

    episode_len = env_results.get(
        "episode_len_mean",
        None
    )

    print(
        f"\nIteration {iteration:03d}"
    )

    print(
        "  reward_mean:",
        episode_reward
    )

    print(
        "  episode_len:",
        episode_len
    )

    if iteration % 10 == 0:

        checkpoint = algo.save(
            CHECKPOINT_DIR
        )

        print(
            "\nCheckpoint saved:"
        )

        print(
            checkpoint
        )


# ============================================================
# FINAL CHECKPOINT
# ============================================================

print("\n==============================================")
print("SAVING FINAL RSAC CHECKPOINT")
print("==============================================")

final_checkpoint = algo.save(
    CHECKPOINT_DIR
)

print(
    "\nFINAL CHECKPOINT:"
)

print(
    final_checkpoint
)

print("\n==============================================")
print("TRAINING COMPLETE")
print("==============================================")


# ============================================================
# CLEAN SHUTDOWN
# ============================================================

algo.stop()

ray.shutdown()