# run_hmarl.py

from env4 import UAVHMARLEnv
from mid import MidLevelPlanner
from sac1 import LowLevelController
from ray.rllib.algorithms.ppo import PPO

import os





# =========================
# INIT
# =========================
env = UAVHMARLEnv()
mid = MidLevelPlanner()
low = LowLevelController(env.agents)   # ✅ FIX 1 (pass agents)

base_dir = "E:/uav_hmarl_project/macro_policy"

algo_dir = os.path.join(base_dir, os.listdir(base_dir)[0])
checkpoints = sorted(os.listdir(algo_dir))

latest_checkpoint = os.path.join(algo_dir, checkpoints[-1])

macro = PPO.from_checkpoint(latest_checkpoint)

obs, _ = env.reset()
low.reset()   # ✅ FIX 2 (reset memory)

logs = []

# =========================
# ROLLOUT
# =========================
for t in range(200):

    # 🔴 HIGH LEVEL (every k steps)
    if t % 10 == 0:
        macro_actions = {
            a: macro.compute_single_action(o)
            for a, o in obs.items()
        }

        # ✅ FIX 3 (convert leader signal → task index)
        leader_assignments = {
            a: int(((macro_actions[a][3] + 1) / 2) * (env.num_tasks - 1))
            for a in obs
        }

    # 🟡 MID LEVEL (planner)
    tasks = env.tasks

    # ✅ FIX 4 (use full pipeline)
    assignments, intents = mid.plan(obs, tasks, leader_assignments)

    # 🟢 LOW LEVEL (RSAC-lite with memory)
    actions = {
        a: low.act(a, obs[a], intents[a])   # ✅ FIX 5 (pass agent id)
        for a in obs
    }

    # STEP
    obs, rewards, terms, truncs, _ = env.step(actions)

    # LOGGING
    logs.append({
        "positions": list(env.pos.values()),
        "tasks": tasks,
        "assignments": assignments,
        "intents": intents
    })

    # TERMINATION
    if all(terms.values()):
        break