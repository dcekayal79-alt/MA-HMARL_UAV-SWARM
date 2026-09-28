import os
import ray
import torch
import numpy as np

from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from rsacenv import UAVRSACEnv

from lstmmodule import (
    RecurrentSACTorchRLModule
)


# =====================================================
# SETTINGS
# =====================================================

CHECKPOINT_DIR = (
    "E:/uav_hmarl_project/"
    "rsac_lstm_test_checkpoint"
)

NUM_ITERATIONS = 3

NUM_ENV_RUNNERS = 1


# =====================================================
# ENVIRONMENT
# =====================================================

def env_creator(config):

    return UAVRSACEnv()


register_env(
    "low_level_env",
    env_creator
)


# =====================================================
# ENV TEST
# =====================================================

print(
    "\n=============================================="
)

print(
    "TESTING ENVIRONMENT"
)

print(
    "=============================================="
)

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

assert (
    test_env.observation_space.shape
    == (13,)
), (
    "Expected 13-D observation"
)

assert (
    test_env.action_space.shape
    == (3,)
), (
    "Expected 3-D action"
)

test_env.close()

print(
    "Environment check PASSED"
)


# =====================================================
# RL MODULE SPEC
# =====================================================

print(
    "\n=============================================="
)

print(
    "BUILDING RECURRENT SAC CONFIG"
)

print(
    "=============================================="
)


# =====================================================
# SAC CONFIG
# =====================================================

config = (

    SACConfig()

    # -------------------------------------------------
    # NEW API STACK
    # -------------------------------------------------
    #
    # We intentionally use the RLModule stack because
    # our custom LSTM is an RLModule.
    #
    # -------------------------------------------------

    .api_stack(

        enable_rl_module_and_learner=True,

        enable_env_runner_and_connector_v2=True
    )

    # -------------------------------------------------
    # ENV
    # -------------------------------------------------

    .environment(

        env="low_level_env"
    )

    # -------------------------------------------------
    # TORCH
    # -------------------------------------------------

    .framework(
        "torch"
    )

    # -------------------------------------------------
    # ENV RUNNERS
    # -------------------------------------------------

    .env_runners(

        num_env_runners=NUM_ENV_RUNNERS
    )

    # -------------------------------------------------
    # RL MODULE
    # -------------------------------------------------

    .rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=RecurrentSACTorchRLModule,
            model_config={
            "fcnet_hiddens": [256, 256],
            "max_seq_len": 20,
            },
        )
    )


    # -------------------------------------------------
    # TRAINING
    # -------------------------------------------------

    .training(

        gamma=0.99,

        train_batch_size=256,

        actor_lr=3e-4,

        critic_lr=3e-4,

        alpha_lr=3e-4,

        tau=0.005,

        replay_buffer_config={

            "capacity": 100000
        }
    )

    # -------------------------------------------------
    # RESOURCES
    # -------------------------------------------------

    .resources(

        num_gpus=0
    )
)


# =====================================================
# BUILD
# =====================================================

print(
    "\n=============================================="
)

print(
    "BUILDING SAC"
)

print(
    "=============================================="
)

algo = config.build_algo()

print(
    "\nSAC BUILD SUCCESSFUL"
)


# =====================================================
# INSPECT MODULE
# =====================================================

print(
    "\n=============================================="
)

print(
    "INSPECTING RL MODULE"
)

print(
    "=============================================="
)

module = algo.get_module()

print(
    "Module type:"
)

print(
    type(module)
)

print(
    "\nModule class:"
)

print(
    module.__class__.__name__
)

print(
    "\nInitial state:"
)

initial_state = (
    module.get_initial_state()
)

print(
    initial_state
)


# =====================================================
# VERIFY INITIAL STATE
# =====================================================

assert isinstance(
    initial_state,
    dict
), (
    "Initial state must be a dictionary"
)

assert "h" in initial_state
assert "c" in initial_state

print(
    "\nInitial state check PASSED"
)

print(
    "h shape:",
    initial_state["h"].shape
)

print(
    "c shape:",
    initial_state["c"].shape
)


# =====================================================
# TRAIN
# =====================================================

print(
    "\n=============================================="
)

print(
    "STARTING LSTM SAC TEST TRAINING"
)

print(
    "Iterations:",
    NUM_ITERATIONS
)

print(
    "=============================================="
)


rewards = []


for i in range(NUM_ITERATIONS):

    print(
        f"\n----------- ITERATION {i} -----------"
    )

    result = algo.train()

    # -------------------------------------------------
    # Extract reward safely.
    # -------------------------------------------------

    reward = None

    try:

        reward = result[
            "env_runners"
        ][
            "episode_return_mean"
        ]

    except Exception:

        pass

    if reward is None:

        try:

            reward = result[
                "episode_reward_mean"
            ]

        except Exception:

            reward = np.nan

    rewards.append(
        reward
    )

    print(
        f"Iteration {i}"
    )

    print(
        f"Episode reward: {reward}"
    )


# =====================================================
# SAVE
# =====================================================

print(
    "\n=============================================="
)

print(
    "SAVING TEST CHECKPOINT"
)

print(
    "=============================================="
)

checkpoint = algo.save(
    CHECKPOINT_DIR
)

print(
    "Checkpoint:"
)

print(
    checkpoint
)


# =====================================================
# FINAL MODULE CHECK
# =====================================================

print(
    "\n=============================================="
)

print(
    "FINAL MODULE CHECK"
)

print(
    "=============================================="
)

module = algo.get_module()

print(
    "Module:",
    type(module)
)

print(
    "Initial state:"
)

print(
    module.get_initial_state()
)


# =====================================================
# SHUTDOWN
# =====================================================

algo.stop()

ray.shutdown()

print(
    "\n=============================================="
)

print(
    "TEST COMPLETE"
)

print(
    "=============================================="
)