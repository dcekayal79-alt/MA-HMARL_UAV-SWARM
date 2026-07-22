import ray
import numpy as np
import matplotlib.pyplot as plt

from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPO

from commonhmarlenv import UAVHMARLEnv
from mid import MidLevelPlanner
from lowlevelcontroller import LowLevelController
from rsacenv import UAVRSACEnv

# =========================================================
# INIT RAY
# =========================================================
ray.init(ignore_reinit_error=True)

# =========================================================
# ENV REGISTRATION
# =========================================================
def env_creator(config):

    return ParallelPettingZooEnv(
        UAVHMARLEnv(num_uavs=5)
    )

register_env("uav_hmarl", env_creator)

# =========================================================
# LOW-LEVEL ENV
# =========================================================
def low_env_creator(config):

    return UAVRSACEnv()

register_env("low_level_env", low_env_creator)

# =========================================================
# CREATE ENVIRONMENT
# =========================================================
env = UAVHMARLEnv(num_uavs=5)

agents = env.agents

# =========================================================
# MACRO RECURRENT STATES (MAPPO LSTM)
# =========================================================
macro_hidden_states = {}


# =========================================================
# PREVIOUS MACRO ACTIONS (INIT)
# =========================================================
prev_macro_actions = {
    a: np.zeros(3, dtype=np.float32)
    for a in agents
}

# =========================================================
# PREVIOUS MACRO REWARDS (INIT)
# =========================================================
prev_macro_rewards = {
    a: 0.0
    for a in agents
}
# =========================================================
# OBSERVATION CHECK
# =========================================================
print("\n====================================")
print("OBSERVATION SPACE CHECK")
print("====================================")

sample_agent = agents[0]

obs_space = env.observation_space(sample_agent)

print(obs_space)

# =========================================================
# MID-LEVEL PLANNER
# =========================================================
mid = MidLevelPlanner()

# =========================================================
# LOW-LEVEL CONTROLLER
# =========================================================
low = LowLevelController(agents)

# =========================================================
# LOAD TRAINED MAPPO POLICY
# =========================================================
macro = PPO.from_checkpoint(
    "E:/uav_hmarl_project/mappo_lstm_final"
)

# =========================================================
# RESET ENVIRONMENT
# =========================================================
obs, infos = env.reset()

# =========================================================
# INITIAL MACRO ACTIONS
# =========================================================
macro_actions = {
    a: 0 for a in agents
}

# =========================================================
# METRICS
# =========================================================
episode_rewards = []

success_history = []

collision_history = []

coverage_history = []

# =========================================================
# TRAJECTORIES
# =========================================================
trajectories = {
    a: [] for a in agents
}

# =========================================================
# COVERAGE TRACKER
# =========================================================
visited_cells = set()

# =========================================================
# LOW-LEVEL RECURRENT STATES
# =========================================================
low_hidden_states = {
    a: None for a in agents
}

# =========================================================
# MACRO RECURRENT STATES (MAPPO LSTM)
# =========================================================
macro_hidden_states = {}

# =========================================================
# PLOT SETUP
# =========================================================
plt.ion()

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(
    111,
    projection="3d"
)

# =========================================================
# MAIN LOOP
# =========================================================
max_steps = 200

for t in range(max_steps):

    print(f"\n================ STEP {t} ================")

    # =====================================================
    # CURRICULUM DIFFICULTY
    # =====================================================
    difficulty = min(t / max_steps, 1.0)

    env.dropout_prob = (
        0.1 + 0.7 * difficulty
    )

      # =====================================================
    # HIGH-LEVEL MAPPO
    # UPDATE EVERY 10 STEPS
    # =====================================================
    if t % 10 == 0:

        macro_actions = {}

        for agent, o in obs.items():

            policy_id = (
                "leader_policy"
                if agent == "uav_0"
                else "worker_policy"
            )

            # =============================================
            # GET POLICY
            # =============================================
            policy = macro.get_policy(policy_id)

            # =============================================
            # INITIAL LSTM STATE
            # =============================================
            if agent not in macro_hidden_states:

                macro_hidden_states[agent] = \
                    policy.get_initial_state()

            # =============================================
            # COMPUTE ACTION
            # =============================================
            action, state_out, _ = macro.compute_single_action(
                observation=o,
                state=macro_hidden_states[agent],
                prev_action=prev_macro_actions[agent],
                prev_reward=prev_macro_rewards[agent],
                policy_id=policy_id,
                explore=False
            )

            # =============================================
            # STORE NEXT STATE
            # =============================================
            macro_hidden_states[agent] = state_out

            # =============================================
            # CLEAN ACTION FORMAT (SAFE)
            # =============================================
            if isinstance(action, np.ndarray):

                 action = np.asarray(action)

            # If action is vector (e.g., Box(3,))
                 action = action.flatten()

            macro_actions[agent] = action

            # =============================================
            # STORE PREVIOUS ACTION
            # =============================================
            prev_macro_actions[agent] = action

        print("\nMacro Actions:")
        print(macro_actions)
    # =====================================================
    # MID-LEVEL PLANNER
    # =====================================================
    assignments, intents = mid.plan(
        observations=obs,
        tasks=env.tasks,
        macro_actions=macro_actions
    )

    print("\nAssignments:")
    print(assignments)

    print("\nIntents:")
    print(intents)

    # =====================================================
    # LOW-LEVEL RSAC CONTROL
    # =====================================================
    actions = {}

    for agent in obs:

        action, next_hidden = low.act(
            agent,
            obs[agent],
            intents[agent],
            low_hidden_states[agent]
        )

        low_hidden_states[agent] = next_hidden

        actions[agent] = action

    print("\nLow-Level Actions:")
    print(actions)

    # =====================================================
    # ENVIRONMENT STEP
    # =====================================================
    obs, rewards, terms, truncs, infos = env.step(actions)

    # =====================================================
    # UPDATE PREVIOUS REWARDS
    # =====================================================
    for agent in rewards:

        prev_macro_rewards[agent] = rewards[agent]

    print("\nRewards:")
    print(rewards)
  

    # =====================================================
    # TOTAL REWARD
    # =====================================================
    total_reward = sum(rewards.values())

    episode_rewards.append(total_reward)

    # =====================================================
    # SUCCESS METRIC
    # =====================================================
    success = terms.get("__all__", False)

    success_history.append(
        1 if success else 0
    )

    # =====================================================
    # COLLISION COUNT
    # =====================================================
    collisions = 0

    for i, a1 in enumerate(agents):

        for j, a2 in enumerate(agents):

            if i >= j:
                continue

            dist = np.linalg.norm(
                env.pos[a1] - env.pos[a2]
            )

            if dist < env.safe_distance:
                collisions += 1

    collision_history.append(collisions)

    # =====================================================
    # COVERAGE METRIC
    # =====================================================
    for agent in agents:

        pos = env.pos[agent]

        cell = (
            int(pos[0]),
            int(pos[1]),
            int(pos[2])
        )

        visited_cells.add(cell)

    coverage_history.append(
        len(visited_cells)
    )

    # =====================================================
    # STORE TRAJECTORIES
    # =====================================================
    for agent, pos in env.pos.items():

        trajectories[agent].append(
            pos.copy()
        )

    # =====================================================
    # UPDATE PLOT
    # =====================================================
    ax.clear()

    for agent, traj in trajectories.items():

        traj = np.array(traj)

        if len(traj) > 0:

            ax.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                label=agent
            )

            ax.scatter(
                traj[-1, 0],
                traj[-1, 1],
                traj[-1, 2]
            )

    # =====================================================
    # TASK VISUALIZATION
    # =====================================================
    tasks_np = np.array(env.tasks)

    ax.scatter(
        tasks_np[:, 0],
        tasks_np[:, 1],
        tasks_np[:, 2],
        c="red",
        marker="x",
        s=100,
        label="Tasks"
    )

    ax.set_title(
        f"Memory-Augmented HMARL | Step {t}"
    )

    ax.set_xlabel("X")

    ax.set_ylabel("Y")

    ax.set_zlabel("Z")

    ax.legend()

    plt.draw()

    plt.pause(0.05)

    # =====================================================
    # TERMINATION
    # =====================================================
    if success:

        print("\nMISSION SUCCESS")

        break

    if truncs.get("__all__", False):

        print("\nTIMEOUT")

        break

# =========================================================
# FINAL VISUALIZATION
# =========================================================
plt.ioff()

# =========================================================
# REWARD CURVE
# =========================================================
plt.figure(figsize=(8, 5))

plt.plot(episode_rewards)

plt.xlabel("Step")

plt.ylabel("Total Reward")

plt.title("HMARL Reward Curve")

plt.grid()

# =========================================================
# SMOOTHED REWARD
# =========================================================
window = 10

if len(episode_rewards) >= window:

    smoothed = np.convolve(
        episode_rewards,
        np.ones(window) / window,
        mode="valid"
    )

    plt.figure(figsize=(8, 5))

    plt.plot(smoothed)

    plt.xlabel("Step")

    plt.ylabel("Smoothed Reward")

    plt.title("Smoothed HMARL Reward")

    plt.grid()

# =========================================================
# COLLISION PLOT
# =========================================================
plt.figure(figsize=(8, 5))

plt.plot(collision_history)

plt.xlabel("Step")

plt.ylabel("Collision Count")

plt.title("Collision Avoidance Performance")

plt.grid()

# =========================================================
# COVERAGE PLOT
# =========================================================
plt.figure(figsize=(8, 5))

plt.plot(coverage_history)

plt.xlabel("Step")

plt.ylabel("Visited Cells")

plt.title("Coverage Performance")

plt.grid()

# =========================================================
# SUCCESS RATE
# =========================================================
plt.figure(figsize=(8, 5))

success_rate = np.cumsum(
    success_history
) / np.arange(
    1,
    len(success_history) + 1
)

plt.plot(success_rate)

plt.xlabel("Step")

plt.ylabel("Success Rate")

plt.title("Mission Success Rate")

plt.grid()

plt.show()

# =========================================================
# FINAL METRICS
# =========================================================
print("\n================ FINAL METRICS ================")

print(f"Total Reward: {np.sum(episode_rewards):.2f}")

print(f"Average Reward: {np.mean(episode_rewards):.2f}")

print(f"Total Collisions: {np.sum(collision_history)}")

print(f"Coverage Cells: {len(visited_cells)}")

print(f"Final Success Rate: {success_rate[-1]:.2f}")

# =========================================================
# SHUTDOWN
# =========================================================
ray.shutdown()