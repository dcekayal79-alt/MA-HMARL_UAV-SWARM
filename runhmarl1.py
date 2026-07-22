import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from env5 import UAVHMARLEnv

# =========================
# INIT RAY
# =========================
ray.init(ignore_reinit_error=True)

# =========================
# ENV REGISTRATION (SAME AS TRAINING)
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
# LOAD TRAINED MACRO POLICY
# =========================
config = (
    PPOConfig()
    .environment("uav_hmarl")
    .framework("torch")
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
    )
)

algo = config.build_algo()

# 🔴 IMPORTANT: use FULL PATH
algo.restore("E:/uav_hmarl_project/macro_policy")

# =========================
# RUN INFERENCE
# =========================
env = env_creator({})

obs, _ = env.reset()

done = {"__all__": False}

while not done["__all__"]:
    actions = {}

    for agent_id, agent_obs in obs.items():
        policy_id = policy_mapping_fn(agent_id)

        action = algo.compute_single_action(
            agent_obs,
            policy_id=policy_id
        )

        actions[agent_id] = action

    obs, rewards, done, truncated, info = env.step(actions)

    print("Rewards:", rewards)

# =========================
# SHUTDOWN
# =========================
ray.shutdown()