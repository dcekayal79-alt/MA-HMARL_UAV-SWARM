import numpy as np
import torch
import ray
import matplotlib.pyplot as plt

from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPO

from env5 import UAVHMARLEnv
from mid import MidLevelPlanner
from sac1 import LowLevelController

# =========================
# INIT RAY + REGISTER ENV
# =========================
ray.init(ignore_reinit_error=True)

def env_creator(config):
    return ParallelPettingZooEnv(UAVHMARLEnv(num_uavs=5))

register_env("uav_hmarl", env_creator)

# =========================
# CREATE ENV + MODULES
# =========================
env = UAVHMARLEnv(num_uavs=5)

mid = MidLevelPlanner()

agents = [f"uav_{i}" for i in range(5)]
low = LowLevelController(agents)

# =========================
# LOAD MACRO POLICY
# =========================
macro = PPO.from_checkpoint("E:/uav_hmarl_project/macro_policy")

leader_module = macro.get_module("leader_policy")
worker_module = macro.get_module("worker_policy")

# =========================
# RESET
# =========================
obs, _ = env.reset()
macro_actions = {a: 0 for a in obs}

# =========================
# 🎯 PLOTTING SETUP
# =========================
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

trajectories = {a: [] for a in agents}

# =========================
# MAIN LOOP
# =========================
for t in range(50):

    print(f"\n===== STEP {t} =====")

    # 🔴 HIGH-LEVEL
    if t % 10 == 0:
        macro_actions = {}

        for agent, o in obs.items():

            module = leader_module if agent == "uav_0" else worker_module

            obs_tensor = torch.tensor(np.array([o]), dtype=torch.float32)

            out = module.forward_inference({"obs": obs_tensor})

            dist_cls = module.get_inference_action_dist_cls()
            dist = dist_cls.from_logits(out["action_dist_inputs"])

            action = dist.to_deterministic().sample()[0].detach().cpu().numpy()

            macro_actions[agent] = action

        print("Macro actions:", macro_actions)

    # 🟡 MID-LEVEL
    tasks = env.tasks

    assignments, intents = mid.plan(
        observations=obs,
        tasks=tasks,
        macro_actions=macro_actions
    )

    print("Assignments:", assignments)
    print("Intents:", intents)

    # 🟢 LOW-LEVEL
    actions = {}

    for agent in obs:
        actions[agent] = low.act(
            agent,
            obs[agent],
            intents[agent]
        )

    print("Low-level actions:", actions)

    # ENV STEP
    obs, rewards, terms, truncs, info = env.step(actions)

    print("Rewards:", rewards)

    # =========================
    # 📊 UPDATE TRAJECTORY PLOT
    # =========================
    for agent, pos in env.pos.items():   # ⚠️ use env.pos (your env uses this)
        trajectories[agent].append(pos)

    ax.clear()

    # plot UAV paths
    for agent, traj in trajectories.items():
        traj = np.array(traj)

        if len(traj) > 0:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=agent)
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2])

    # plot tasks
    tasks_np = np.array(env.tasks)
    ax.scatter(tasks_np[:, 0], tasks_np[:, 1], tasks_np[:, 2],
               c='red', marker='x', s=100, label='Tasks')

    ax.set_title(f"Step {t}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.draw()
    plt.pause(0.1)

    # =========================
    # TERMINATION
    # =========================
    if all(terms.values()):
        print("Episode finished")
        break

# =========================
# CLEANUP
# =========================
plt.ioff()
plt.show()

ray.shutdown()