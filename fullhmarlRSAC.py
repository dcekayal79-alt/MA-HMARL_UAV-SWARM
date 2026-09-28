import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

import ray

from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

from commonhmarlenv import UAVHMARLEnv
from rsacenv import UAVRSACEnv
from mid import MidLevelPlanner
from lowlevelcontrollernew1 import LowLevelController


# ============================================================
# GLOBAL SEED
# ============================================================

SEED = 2024


# ============================================================
# SEED FUNCTION
# ============================================================

def set_global_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# PETTINGZOO ENV CREATOR FOR RLlib CHECKPOINT
# ============================================================

def hmarl_rllib_env_creator(config):

    env = UAVHMARLEnv(
        num_uavs=5,
        num_tasks=5,
        curriculum_level=0
    )

    return PettingZooEnv(env)


# ============================================================
# DIRECT ENV CREATOR
# ============================================================

def hmarl_direct_env_creator(config):

    return UAVHMARLEnv(
        num_uavs=5,
        num_tasks=5,
        curriculum_level=0
    )


# ============================================================
# RSAC ENV CREATOR
# ============================================================

def rsac_env_creator(config):

    return UAVRSACEnv()


# ============================================================
# SAFE STATE COPY
# ============================================================

def copy_state(state):

    if state is None:
        return None

    if isinstance(state, dict):

        return {
            k: copy_state(v)
            for k, v in state.items()
        }

    if isinstance(state, list):

        return [
            copy_state(v)
            for v in state
        ]

    if isinstance(state, tuple):

        return tuple(
            copy_state(v)
            for v in state
        )

    if torch.is_tensor(state):

        return state.clone()

    if isinstance(state, np.ndarray):

        return state.copy()

    return state


# ============================================================
# STATE SHAPES
# ============================================================

def state_shapes(state):

    if state is None:
        return None

    if isinstance(state, dict):

        return {
            k: state_shapes(v)
            for k, v in state.items()
        }

    if isinstance(state, (list, tuple)):

        return [
            state_shapes(v)
            for v in state
        ]

    if torch.is_tensor(state):

        return tuple(state.shape)

    if isinstance(state, np.ndarray):

        return tuple(state.shape)

    return type(state).__name__


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==============================================")
    print("        MEMORY-AWARE HMARL EVALUATION")
    print("==============================================")

    print(
        f"Seed = {SEED}"
    )

    # --------------------------------------------------------
    # GLOBAL SEED
    # --------------------------------------------------------

    set_global_seed(SEED)

    # --------------------------------------------------------
    # RAY
    # --------------------------------------------------------

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False
    )

    # ========================================================
    # REGISTER ENVIRONMENTS
    # ========================================================

    print("\n==============================================")
    print("REGISTERING ENVIRONMENTS")
    print("==============================================")

    register_env(
        "memory_aware_hmarl_env_v1",
        hmarl_rllib_env_creator
    )

    register_env(
        "uav_hmarl",
        hmarl_rllib_env_creator
    )

    register_env(
        "UAV_HMARL_ENV",
        hmarl_rllib_env_creator
    )

    register_env(
        "uav_rsac",
        rsac_env_creator
    )

    register_env(
        "UAV_RSAC_ENV",
        rsac_env_creator
    )

    print(
        "[ENV] RLlib PettingZoo wrapper registered"
    )

    # ========================================================
    # DIRECT HMARL ENV
    # ========================================================

    env = UAVHMARLEnv(
        num_uavs=5,
        num_tasks=5,
        curriculum_level=0
    )

    agents = env.possible_agents[:]

    print("\n==============================================")
    print("HMARL ENVIRONMENT")
    print("==============================================")

    print(
        "Agents       :",
        agents
    )

    print(
        "Num UAVs     :",
        env.num_uavs
    )

    print(
        "Num Tasks    :",
        env.num_tasks
    )

    print(
        "World size   :",
        env.world_size
    )

    print(
        "Max steps    :",
        env.max_steps
    )

    print(
        "Comm radius  :",
        env.r_comm
    )

    print(
        "Sensor range :",
        env.sensor_range
    )

    # ========================================================
    # LOAD MAPPO
    # ========================================================

    print("\n==============================================")
    print("LOADING RECURRENT MAPPO")
    print("==============================================")

    mappo_checkpoint = (
        "E:/uav_hmarl_project/mappo_lstm_final"
    )

    print(
        "MAPPO checkpoint:",
        mappo_checkpoint
    )

    macro = PPO.from_checkpoint(
        mappo_checkpoint
    )

    print(
        "\n[MAPPO] Checkpoint loaded successfully"
    )

    # ========================================================
    # GET POLICIES
    # ========================================================

    leader_policy = macro.get_policy(
        "leader_policy"
    )

    worker_policy = macro.get_policy(
        "worker_policy"
    )

    print(
        "[MAPPO] leader_policy loaded"
    )

    print(
        "[MAPPO] worker_policy loaded"
    )

    # ========================================================
    # INITIAL MAPPO RECURRENT STATES
    # ========================================================

    leader_initial_state = (
        leader_policy.get_initial_state()
    )

    worker_initial_state = (
        worker_policy.get_initial_state()
    )

    print(
        "\n[MAPPO] Leader initial state:",
        state_shapes(
            leader_initial_state
        )
    )

    print(
        "[MAPPO] Worker initial state:",
        state_shapes(
            worker_initial_state
        )
    )

    macro_hidden_states = {}

    for agent in agents:

        if agent == "uav_0":

            macro_hidden_states[agent] = (
                copy_state(
                    leader_initial_state
                )
            )

        else:

            macro_hidden_states[agent] = (
                copy_state(
                    worker_initial_state
                )
            )

    # ========================================================
    # PREVIOUS MAPPO INPUTS
    # ========================================================

    prev_macro_actions = {

        agent: np.zeros(
            3,
            dtype=np.float32
        )

        for agent in agents
    }

    prev_macro_rewards = {

        agent: 0.0

        for agent in agents
    }

    # ========================================================
    # MID-LEVEL
    # ========================================================

    mid = MidLevelPlanner(
        refine=True,
        bias_strength=1.0
    )

    print(
        "\n[MID] Mid-level planner initialized"
    )

    # ========================================================
    # LOW-LEVEL RSAC
    # ========================================================

    print(
        "\n=============================================="
    )

    print(
        "LOADING RECURRENT SAC LOW-LEVEL CONTROLLER"
    )

    print(
        "=============================================="
    )

    low = LowLevelController(
        agents
    )

    print(
        "[LOW] Low-level controller initialized"
    )

    # ========================================================
    # RESET ENVIRONMENT
    # ========================================================

    obs, infos = env.reset(
        seed=SEED
    )

    print(
        "\n=============================================="
    )

    print(
        "ENVIRONMENT RESET"
    )

    print(
        "=============================================="
    )

    print(
        "Initial task assignments:",
        env.task_assignments
    )

    print(
        "Initial tasks:"
    )

    for i, task in enumerate(
        env.tasks
    ):

        print(
            f"  Task {i}: "
            f"{np.round(task, 3)}"
        )

    # ========================================================
    # INITIAL LOW-LEVEL STATES
    # ========================================================

    low_hidden_states = {}

    for agent in agents:

        low_hidden_states[agent] = (
            copy_state(
                low.hidden_states[agent]
            )
        )

    # ========================================================
    # METRICS
    # ========================================================

    total_reward = 0.0

    total_collisions = 0

    total_coverage = 0

    total_successful_uavs = 0

    total_path_length = 0.0

    previous_collision_count = 0

    previous_positions = {

        agent: env.pos[agent].copy()

        for agent in agents
    }

    successful_agents = set()

    trajectory_history = {

        agent: [
            env.pos[agent].copy()
        ]

        for agent in agents
    }

    # ========================================================
    # MAIN EPISODE LOOP
    # ========================================================

    max_steps = 200

    step_idx = -1

    for step_idx in range(
        max_steps
    ):

        print(
            "\n================================================"
        )

        print(
            f"                    STEP {step_idx + 1}"
        )

        print(
            "================================================"
        )

        # ====================================================
        # 1. MAPPO
        # ====================================================

        macro_actions = {}

        for agent in agents:

            if agent == "uav_0":

                policy = leader_policy

            else:

                policy = worker_policy

            state_in = copy_state(
                macro_hidden_states[agent]
            )

            prev_action = np.asarray(
                prev_macro_actions[agent],
                dtype=np.float32
            ).reshape(3)

            prev_reward = float(
                prev_macro_rewards[agent]
            )

            result = (
                policy.compute_single_action(
                    obs=obs[agent],
                    state=state_in,
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                    explore=False
                )
            )

            macro_action = result[0]

            state_out = result[1]

            macro_action = np.asarray(
                macro_action,
                dtype=np.float32
            ).reshape(-1)

            if macro_action.size != 3:

                raise RuntimeError(
                    f"[MAPPO ERROR] {agent}: "
                    f"Expected 3-D macro action, "
                    f"got {macro_action.shape}"
                )

            macro_actions[agent] = (
                macro_action.copy()
            )

            macro_hidden_states[agent] = (
                copy_state(
                    state_out
                )
            )

            print(
                f"[MAPPO] {agent} | "
                f"action="
                f"{np.round(macro_action, 4)} | "
                f"state="
                f"{state_shapes(state_out)}"
            )

        # ====================================================
        # 2. MID-LEVEL
        # ====================================================

        assignments, intents = mid.plan(
            obs,
            env.tasks,
            macro_actions
        )

        print(
            "\n[MID] Task assignments"
        )

        for agent in agents:

            assigned_task = np.asarray(
                assignments[agent],
                dtype=np.float32
            ).reshape(-1)

            if assigned_task.size != 3:

                raise RuntimeError(
                    f"[MID ERROR] {agent}: "
                    f"Expected 3-D task, "
                    f"got {assigned_task.shape}"
                )

            print(
                f"[MID] {agent} "
                f"assigned_task = "
                f"{np.round(assigned_task, 3)}"
            )

        # ====================================================
        # 3. UPDATE ENVIRONMENT TASK ASSIGNMENTS
        # ====================================================

        for agent in agents:

            assigned_task = np.asarray(
                assignments[agent],
                dtype=np.float32
            ).reshape(-1)

            task_distances = np.linalg.norm(
                env.tasks -
                assigned_task,
                axis=1
            )

            task_idx = int(
                np.argmin(
                    task_distances
                )
            )

            env.task_assignments[agent] = (
                task_idx
            )

            actual_task = (
                env.tasks[task_idx]
                .copy()
                .astype(np.float32)
            )

            assignment_error = np.linalg.norm(
                actual_task -
                assigned_task
            )

            print(
                f"\n[TASK LINK] {agent}"
            )

            print(
                f"    Mid-level task      = "
                f"{np.round(assigned_task, 3)}"
            )

            print(
                f"    Selected task index = "
                f"{task_idx}"
            )

            print(
                f"    Environment task    = "
                f"{np.round(actual_task, 3)}"
            )

            print(
                f"    Assignment error    = "
                f"{assignment_error:.6f}"
            )

        # ====================================================
        # 4. RSAC
        # ====================================================

        rsac_actions = {}

        print(
            "\n=============================================="
        )

        print(
            "LOW-LEVEL RSAC"
        )

        print(
            "=============================================="
        )

        for agent in agents:

            # ------------------------------------------------
            # RSAC observation
            # ------------------------------------------------

            rsac_obs = env._rsac_obs(
                agent
            )

            rsac_obs = np.asarray(
                rsac_obs,
                dtype=np.float32
            )

            if rsac_obs.shape != (13,):

                raise RuntimeError(
                    f"[RSAC ERROR] {agent}: "
                    f"Expected observation shape "
                    f"(13,), got "
                    f"{rsac_obs.shape}"
                )
            position = rsac_obs[0:3]
            target = rsac_obs[6:9]

        target_direction = target - position
        target_norm = np.linalg.norm(target_direction)
        if target_norm > 1e-8:
            target_direction = target_direction / target_norm

            print(
                f"[DIRECTION CHECK] {agent}\n"
                f"    Position        = {np.round(position, 3)}\n"
                f"    Target          = {np.round(target, 3)}\n"
                f"    Target direction= {np.round(target_direction, 3)}"
            )

            # ------------------------------------------------
            # Current assigned task
            # ------------------------------------------------

            task_idx = int(
                env.task_assignments[agent]
            )

            actual_task = (
                env.tasks[task_idx]
            )

            # ------------------------------------------------
            # Expected normalized target
            # ------------------------------------------------

            expected_target = (
                actual_task /
                float(env.world_size)
            )

            expected_target = np.clip(
                expected_target,
                -1.0,
                1.0
            ).astype(np.float32)

            # ------------------------------------------------
            # Target from RSAC observation
            # ------------------------------------------------

            actual_rsac_target = (
                rsac_obs[6:9].copy()
            )

            target_error = np.linalg.norm(
                actual_rsac_target -
                expected_target
            )

            print(
                f"\n[TARGET CHECK] {agent}"
            )

            print(
                f"    Environment task    = "
                f"{np.round(actual_task, 3)}"
            )

            print(
                f"    Expected normalized = "
                f"{np.round(expected_target, 3)}"
            )

            print(
                f"    RSAC target         = "
                f"{np.round(actual_rsac_target, 3)}"
            )

            print(
                f"    Target error        = "
                f"{target_error:.6f}"
            )

            if target_error > 1e-5:

                raise RuntimeError(
                    f"[TARGET ERROR] {agent}: "
                    f"RSAC target does not match "
                    f"environment task."
                )

            # ------------------------------------------------
            # RSAC
            # ------------------------------------------------

            action, next_hidden = low.act(
                agent,
                rsac_obs,
                intents[agent],
                low_hidden_states[agent]
            )

            action = np.asarray(
                action,
                dtype=np.float32
            ).reshape(3)

            rsac_actions[agent] = (
                action.copy()
            )

            low_hidden_states[agent] = (
                copy_state(
                    next_hidden
                )
            )

            print(
                f"[RSAC] {agent} action = "
                f"{np.round(action, 6)}"
            )

        # ====================================================
        # 5. ENVIRONMENT STEP
        # ====================================================

        (
            next_obs,
            rewards,
            terminations,
            truncations,
            infos
        ) = env.step(
            rsac_actions
        )

        # ====================================================
        # 6. PREVIOUS MAPPO INPUTS
        # ====================================================

        for agent in agents:

            prev_macro_actions[agent] = (
                macro_actions[agent].copy()
            )

            prev_macro_rewards[agent] = float(
                rewards.get(
                    agent,
                    0.0
                )
            )

        # ====================================================
        # 7. REWARD
        # ====================================================

        # Preserve the original evaluation convention:
        # use the first available environment reward as
        # the episode-level step reward.

        if len(rewards) > 0:

            step_reward = float(
                next(
                    iter(
                        rewards.values()
                    )
                )
            )

        else:

            step_reward = 0.0

        total_reward += step_reward

        # ----------------------------------------------------
        # Optional diagnostic: all UAV rewards
        # ----------------------------------------------------

        print(
            "\n[REWARD DETAIL]"
        )

        for agent in agents:

            print(
                f"    {agent}: "
                f"{float(rewards.get(agent, 0.0)):.4f}"
            )

        # ====================================================
        # 8. COLLISIONS
        # ====================================================

        current_collision_count = int(
            getattr(
                env,
                "collision_count",
                0
            )
        )

        collision_increment = (
            current_collision_count -
            previous_collision_count
        )

        if collision_increment < 0:

            collision_increment = 0

        total_collisions += (
            collision_increment
        )

        previous_collision_count = (
            current_collision_count
        )

        # ====================================================
        # 9. PATH LENGTH
        # ====================================================

        step_path_length = 0.0

        for agent in agents:

            current_position = (
                env.pos[agent].copy()
            )

            movement = np.linalg.norm(
                current_position -
                previous_positions[agent]
            )

            step_path_length += movement

            total_path_length += movement

            previous_positions[agent] = (
                current_position.copy()
            )

            trajectory_history[agent].append(
                current_position.copy()
            )

        # ====================================================
        # 10. SUCCESS / COVERAGE
        # ====================================================

        step_successful_uavs = 0

        for agent in agents:

            task_idx = int(
                env.task_assignments[agent]
            )

            target = (
                env.tasks[task_idx]
            )

            distance_to_task = np.linalg.norm(
                env.pos[agent] -
                target
            )

            if distance_to_task < 0.5:

                if agent not in successful_agents:

                    successful_agents.add(
                        agent
                    )

                    step_successful_uavs += 1

        # Unique successful UAVs
        total_successful_uavs = len(
            successful_agents
        )

        # Coverage is also the number of unique
        # UAVs that successfully reached their target.
        total_coverage = len(
            successful_agents
        )

        # ====================================================
        # 11. PRINT METRICS
        # ====================================================

        print(
            "\n[REWARD] "
            f"Step reward = "
            f"{step_reward:.4f}"
        )

        print(
            "[REWARD] "
            f"Total reward = "
            f"{total_reward:.4f}"
        )

        print(
            "[SAFETY] "
            f"Step collisions = "
            f"{collision_increment}"
        )

        print(
            "[SAFETY] "
            f"Total collisions = "
            f"{total_collisions}"
        )

        print(
            "[COVERAGE] "
            f"Step new successful UAVs = "
            f"{step_successful_uavs}"
        )

        print(
            "[COVERAGE] "
            f"Total successful UAVs = "
            f"{total_coverage}"
        )

        print(
            "[SUCCESS] "
            f"Step successful UAVs = "
            f"{step_successful_uavs}"
        )

        print(
            "[PATH] "
            f"Step path length = "
            f"{step_path_length:.4f}"
        )

        # ====================================================
        # 12. NEXT OBSERVATION
        # ====================================================

        obs = next_obs

        # ====================================================
        # 13. TERMINATION / TRUNCATION
        # ====================================================

        all_terminated = bool(
            terminations.get(
                "__all__",
                False
            )
        )

        all_truncated = bool(
            truncations.get(
                "__all__",
                False
            )
        )

        if all_terminated:

            print(
                "\nEnvironment signalled "
                "episode termination."
            )

            break

        if all_truncated:

            print(
                "\nEnvironment reached "
                "the episode time/step limit."
            )

            break

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    steps_completed = (
        step_idx + 1
    )

    average_step_reward = (
        total_reward /
        max(
            steps_completed,
            1
        )
    )

    success_rate = (
        total_successful_uavs /
        float(len(agents)) * 100.0
    )

    # ========================================================
    # FINAL UAV POSITION / TARGET DIAGNOSTICS
    # ========================================================

    print(
        "\n=============================================="
    )

    print(
        "FINAL UAV POSITION / TARGET DIAGNOSTICS"
    )

    print(
        "=============================================="
    )

    final_target_distances = {}

    for agent in agents:

        task_idx = int(
            env.task_assignments[agent]
        )

        final_position = np.asarray(
            env.pos[agent],
            dtype=np.float32
        )

        target_position = np.asarray(
            env.tasks[task_idx],
            dtype=np.float32
        )

        final_distance = float(
            np.linalg.norm(
                final_position -
                target_position
            )
        )

        final_target_distances[agent] = (
            final_distance
        )

        print(
            f"\n[{agent}]"
        )

        print(
            f"    Assigned task index : "
            f"{task_idx}"
        )

        print(
            f"    Final position      : "
            f"{np.round(final_position, 3)}"
        )

        print(
            f"    Target position     : "
            f"{np.round(target_position, 3)}"
        )

        print(
            f"    Final target dist.  : "
            f"{final_distance:.6f}"
        )

        if final_distance < 0.5:

            print(
                "    Status              : SUCCESS"
            )

        else:

            print(
                "    Status              : NOT REACHED"
            )

    # ========================================================
    # FINAL DISTANCE SUMMARY
    # ========================================================

    if len(final_target_distances) > 0:

        mean_final_distance = float(
            np.mean(
                list(
                    final_target_distances.values()
                )
            )
        )

        min_final_distance = float(
            np.min(
                list(
                    final_target_distances.values()
                )
            )
        )

        max_final_distance = float(
            np.max(
                list(
                    final_target_distances.values()
                )
            )
        )

    else:

        mean_final_distance = 0.0

        min_final_distance = 0.0

        max_final_distance = 0.0

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n\n================================================"
    )

    print(
        "             HMARL EPISODE COMPLETED"
    )

    print(
        "================================================"
    )

    print(
        "\n=============================================="
    )

    print(
        "FINAL HMARL RESULTS"
    )

    print(
        "=============================================="
    )

    print(
        f"Seed                    : "
        f"{SEED}"
    )

    print(
        f"Steps                   : "
        f"{steps_completed}"
    )

    print(
        f"Total Reward            : "
        f"{total_reward:.4f}"
    )

    print(
        f"Average Step Reward     : "
        f"{average_step_reward:.4f}"
    )

    print(
        f"Total Collisions        : "
        f"{total_collisions}"
    )

    print(
        f"Total Coverage          : "
        f"{total_coverage}"
    )

    print(
        f"Total Successful UAVs   : "
        f"{total_successful_uavs}"
    )

    print(
        f"Success Rate            : "
        f"{success_rate:.4f}"
    )

    print(
        f"Total Path Length       : "
        f"{total_path_length:.4f}"
    )

    print(
        f"Mean Final Target Dist. : "
        f"{mean_final_distance:.6f}"
    )

    print(
        f"Minimum Final Distance  : "
        f"{min_final_distance:.6f}"
    )

    print(
        f"Maximum Final Distance  : "
        f"{max_final_distance:.6f}"
    )

    # ========================================================
    # TRAJECTORY PLOT
    # ========================================================

    print(
        "\nGenerating trajectory plot..."
    )

    try:

        plt.figure(
            figsize=(10, 8)
        )

        # ----------------------------------------------------
        # UAV TRAJECTORIES
        # ----------------------------------------------------

        for agent in agents:

            trajectory = np.asarray(
                trajectory_history[agent]
            )

            if trajectory.ndim != 2:
                continue

            if trajectory.shape[0] == 0:
                continue

            plt.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                label=agent
            )

            plt.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                marker="o"
            )

            plt.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                marker="x"
            )

        # ----------------------------------------------------
        # TASKS
        # ----------------------------------------------------

        for idx, task in enumerate(
            env.tasks
        ):

            plt.scatter(
                task[0],
                task[1],
                marker="*",
                s=120
            )

            plt.text(
                task[0],
                task[1],
                f"T{idx}",
                fontsize=9
            )

        # ----------------------------------------------------
        # OBSTACLES
        # ----------------------------------------------------

        for obstacle in env.obstacles:

            plt.scatter(
                obstacle[0],
                obstacle[1],
                marker="s",
                s=80
            )

        plt.xlabel(
            "X Position"
        )

        plt.ylabel(
            "Y Position"
        )

        plt.title(
            "HMARL UAV Trajectories - Seed 42"
        )

        plt.legend()

        plt.grid(
            True
        )

        plt.tight_layout()

        plot_path = (
            "E:/uav_hmarl_project/"
            "hmarl_trajectory_seed{seed}.png"
        )

        plt.savefig(
            plot_path,
            dpi=200
        )

        plt.close()

        print(
            "Trajectory plot saved to:"
        )

        print(
            plot_path
        )

    except Exception as exc:

        print(
            "\n[WARNING] "
            "Trajectory plotting failed:"
        )

        print(
            repr(exc)
        )

    # ========================================================
    # SHUTDOWN
    # ========================================================

    print(
        "\nShutting down MAPPO/Ray..."
    )

    try:

        macro.stop()

    except Exception:

        pass

    try:

        ray.shutdown()

    except Exception:

        pass

    print(
        "\n=============================================="
    )

    print(
        "PROGRAM FINISHED"
    )

    print(
        "==============================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

