# ============================================================
# abl.py
#
# MA-HMARL ABLATION:
# WITHOUT MID-LEVEL
#
# Architecture:
#
#       Recurrent MAPPO
#             |
#             | strategic action
#             v
#       Recurrent / SAC
#             |
#             | low-level action
#             v
#          UAV Env
#
# MAPPO decision:
#       every 10 environment steps
#
# RSAC decision:
#       every environment step
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np
import torch
import ray
import matplotlib.pyplot as plt

from ray.tune.registry import register_env

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.sac import SAC

from ray.rllib.env.wrappers.pettingzoo_env import (
    ParallelPettingZooEnv
)

from ray.rllib.core.columns import Columns

from commonhmarlenv import UAVHMARLEnv
from rsacenv import UAVRSACEnv


# ============================================================
# ENVIRONMENT SETTINGS
# ============================================================

NUM_UAVS = 5
NUM_TASKS = 5


# ============================================================
# ENVIRONMENT REGISTRATION
# ============================================================

# ------------------------------------------------------------
# MAPPO / HMARL ENVIRONMENT
# ------------------------------------------------------------

def hmarl_env_creator(config):

    pettingzoo_env = UAVHMARLEnv(
        num_uavs=NUM_UAVS,
        num_tasks=NUM_TASKS
    )

    return ParallelPettingZooEnv(
        pettingzoo_env
    )


register_env(
    "uav_hmarl",
    hmarl_env_creator
)


# ------------------------------------------------------------
# RSAC / LOW-LEVEL ENVIRONMENT
# ------------------------------------------------------------

register_env(
    "low_level_env",
    lambda config: UAVRSACEnv()
)


# ============================================================
# RAY INITIALIZATION
# ============================================================

ray.init(
    ignore_reinit_error=True
)


# ============================================================
# CHECKPOINTS
# ============================================================

MAPPO_CHECKPOINT = (
    "E:/uav_hmarl_project/mappo_lstm_final"
)

RSAC_CHECKPOINT = (
    "E:/uav_hmarl_project/sac_checkpoint_final"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

RESULT_DIR = (
    "E:/uav_hmarl_project/ablation_results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "without_mid_level_results.npz"
)

REWARD_PLOT = os.path.join(
    RESULT_DIR,
    "without_mid_level_reward.png"
)

COLLISION_PLOT = os.path.join(
    RESULT_DIR,
    "without_mid_level_collisions.png"
)

COVERAGE_PLOT = os.path.join(
    RESULT_DIR,
    "without_mid_level_coverage.png"
)


# ============================================================
# LOAD MAPPO
# ============================================================

print("\n======================================")
print("Loading MAPPO checkpoint...")
print("======================================")

mappo = PPO.from_checkpoint(
    MAPPO_CHECKPOINT
)

print(
    "MAPPO checkpoint loaded successfully."
)


# ============================================================
# LOAD RSAC
# ============================================================

print("\n======================================")
print("Loading RSAC checkpoint...")
print("======================================")

sac = SAC.from_checkpoint(
    RSAC_CHECKPOINT
)

print(
    "RSAC checkpoint loaded successfully."
)


# ============================================================
# GET MAPPO POLICIES
# ============================================================

print("\nGetting MAPPO policies...")

leader_policy = mappo.get_policy(
    "leader_policy"
)

worker_policy = mappo.get_policy(
    "worker_policy"
)

print(
    "Leader policy:",
    type(leader_policy)
)

print(
    "Worker policy:",
    type(worker_policy)
)


# ============================================================
# GET RSAC RL MODULE
#
# IMPORTANT:
#
# SAC checkpoint is using RLlib's NEW API STACK.
#
# Therefore:
#
#     sac.compute_single_action()
#
# is NOT used.
#
# Instead:
#
#     sac.get_module()
#     module.forward_inference()
#
# ============================================================

print("\nGetting RSAC RLModule...")

sac_module = sac.get_module()

print(
    "RSAC RLModule:",
    type(sac_module)
)


# ============================================================
# CREATE HMARL ENVIRONMENT
# ============================================================

print("\nCreating UAV HMARL environment...")

env = UAVHMARLEnv(
    num_uavs=NUM_UAVS,
    num_tasks=NUM_TASKS
)

agents = env.agents[:]

print(
    "Agents:",
    agents
)


# ============================================================
# RESET ENVIRONMENT
# ============================================================

reset_result = env.reset()


if isinstance(
    reset_result,
    tuple
):

    obs, infos = reset_result

else:

    obs = reset_result
    infos = {}


print(
    "\nEnvironment reset successfully."
)


# ============================================================
# RECURRENT MAPPO STATES
#
# DO NOT initialize these with [].
#
# The MAPPO policies are recurrent.
# RLlib needs the correct initial recurrent state.
# ============================================================

macro_states = {}


for agent in agents:

    if agent == "uav_0":

        macro_states[agent] = (
            leader_policy.get_initial_state()
        )

    else:

        macro_states[agent] = (
            worker_policy.get_initial_state()
        )


# ============================================================
# RSAC RECURRENT STATES
#
# If the loaded SAC RLModule has recurrent state,
# initialize it.
#
# If it is feed-forward SAC, state is simply None.
# ============================================================

low_states = {}


try:

    sac_initial_state = (
        sac_module.get_initial_state()
    )

    print(
        "\nRSAC initial recurrent state detected."
    )

    for agent in agents:

        # Make an independent copy for every UAV
        low_states[agent] = [
            np.copy(x)
            if isinstance(x, np.ndarray)
            else x
            for x in sac_initial_state
        ]

except Exception as e:

    print(
        "\nRSAC appears to be feed-forward."
    )

    print(
        "Reason:",
        e
    )

    for agent in agents:

        low_states[agent] = None


# ============================================================
# PREVIOUS MAPPO ACTIONS
# ============================================================

prev_macro_actions = {

    agent: np.zeros(
        3,
        dtype=np.float32
    )

    for agent in agents

}


# ============================================================
# PREVIOUS MAPPO REWARDS
# ============================================================

prev_macro_rewards = {

    agent: 0.0

    for agent in agents

}


# ============================================================
# MACRO ACTIONS
# ============================================================

macro_actions = {

    agent: np.zeros(
        3,
        dtype=np.float32
    )

    for agent in agents

}


# ============================================================
# METRICS
# ============================================================

episode_rewards = []

collision_history = []

coverage_history = []

success_history = []

path_length_history = []

task_conflict_history = []


# ============================================================
# TRAJECTORIES
# ============================================================

trajectories = {

    agent: []

    for agent in agents

}


# ============================================================
# VISITED CELLS
# ============================================================

visited_cells = set()


# ============================================================
# HELPER:
# BUILD RSAC OBSERVATION
# ============================================================

def build_rsac_observation(
    agent,
    macro_action
):

    """
    Construct the original 13-D RSAC observation.

    RSAC training observation:

        position       3
        velocity       3
        target/task    3
        obstacle vec   3
        obstacle dist  1

        TOTAL = 13

    UAVHMARLEnv does not contain explicit obstacles.

    Therefore:

        nearest obstacle vector = [0, 0, 0]
        nearest obstacle distance = 1

    """

    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

    position = np.asarray(
        env.pos[agent],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # VELOCITY
    # --------------------------------------------------------

    velocity = np.asarray(
        env.vel[agent],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # CURRENT TASK
    # --------------------------------------------------------

    task_index = env.task_assignments[
        agent
    ]

    task = np.asarray(
        env.tasks[task_index],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # NO EXPLICIT OBSTACLE MODEL
    # --------------------------------------------------------

    nearest_obstacle_vector = np.zeros(
        3,
        dtype=np.float32
    )

    nearest_obstacle_distance = 1.0


    # --------------------------------------------------------
    # BUILD 13-D OBSERVATION
    # --------------------------------------------------------

    rsac_obs = np.concatenate([

        position,

        velocity,

        task,

        nearest_obstacle_vector,

        np.array(
            [nearest_obstacle_distance],
            dtype=np.float32
        )

    ]).astype(
        np.float32
    )


    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    rsac_obs[0:3] /= env.world_size

    rsac_obs[3:6] /= 1.0

    rsac_obs[6:9] /= env.world_size

    rsac_obs[9:12] /= 6.0


    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------

    rsac_obs = np.clip(
        rsac_obs,
        -1.0,
        1.0
    )


    return rsac_obs.astype(
        np.float32
    )


# ============================================================
# HELPER:
# COPY RECURRENT STATE
# ============================================================

def copy_state(
    state
):

    if state is None:

        return None


    if isinstance(
        state,
        list
    ):

        return [

            np.copy(x)
            if isinstance(
                x,
                np.ndarray
            )
            else x

            for x in state

        ]


    if isinstance(
        state,
        tuple
    ):

        return tuple(

            np.copy(x)
            if isinstance(
                x,
                np.ndarray
            )
            else x

            for x in state

        )


    if isinstance(
        state,
        np.ndarray
    ):

        return np.copy(
            state
        )


    return state


# ============================================================
# HELPER:
# GET MAPPO ACTION
# ============================================================

def get_mappo_action(
    agent,
    observation
):

    """
    Recurrent MAPPO strategic action.

    uav_0:
        leader_policy

    uav_1..uav_4:
        worker_policy

    MAPPO recurrent state is maintained independently
    for every UAV.
    """

    # --------------------------------------------------------
    # SELECT POLICY
    # --------------------------------------------------------

    if agent == "uav_0":

        policy_id = "leader_policy"

    else:

        policy_id = "worker_policy"


    # --------------------------------------------------------
    # CURRENT RECURRENT STATE
    # --------------------------------------------------------

    state_in = macro_states[agent]


    # --------------------------------------------------------
    # MAPPO INFERENCE
    #
    # This uses the old compatibility API because the
    # loaded MAPPO recurrent checkpoint is currently
    # being evaluated through its Policy interface.
    #
    # The critical fix is that a valid recurrent state
    # is supplied.
    # --------------------------------------------------------

    result = mappo.compute_single_action(

        observation=observation,

        state=state_in,

        prev_action=prev_macro_actions[agent],

        prev_reward=prev_macro_rewards[agent],

        policy_id=policy_id,

        explore=False

    )


    # --------------------------------------------------------
    # EXTRACT ACTION
    # --------------------------------------------------------

    action = result[0]

    state_out = result[1]


    # --------------------------------------------------------
    # UPDATE RECURRENT STATE
    # --------------------------------------------------------

    macro_states[agent] = state_out


    # --------------------------------------------------------
    # CONVERT ACTION
    # --------------------------------------------------------

    action = np.asarray(
        action,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # ENSURE 3-D ACTION
    # --------------------------------------------------------

    if action.size < 3:

        padded = np.zeros(
            3,
            dtype=np.float32
        )

        padded[
            :action.size
        ] = action

        action = padded

    else:

        action = action[:3]


    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------

    action = np.clip(
        action,
        -1.0,
        1.0
    )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    macro_actions[agent] = action

    prev_macro_actions[agent] = (
        action.copy()
    )


    return action


# ============================================================
# HELPER:
# EXTRACT SAC ACTION
# ============================================================

def extract_sac_action(
    result
):

    """
    Extract action from the RLModule inference output.

    Different RLlib versions/checkpoints may expose the
    action using slightly different output structures.

    Preferred key:
        Columns.ACTIONS

    Fallback:
        'actions'
    """

    # --------------------------------------------------------
    # PRINT KEYS ONLY ON FIRST CALL
    # --------------------------------------------------------

    if not hasattr(
        extract_sac_action,
        "printed_keys"
    ):

        print(
            "\nRSAC RLModule output keys:",
            list(result.keys())
        )

        extract_sac_action.printed_keys = True


    # --------------------------------------------------------
    # NORMAL ACTION OUTPUT
    # --------------------------------------------------------

    if Columns.ACTIONS in result:

        action = result[
            Columns.ACTIONS
        ]

    elif "actions" in result:

        action = result[
            "actions"
        ]

    # --------------------------------------------------------
    # SOME RLlib MODULES MAY RETURN DISTRIBUTION INPUTS
    # --------------------------------------------------------

    elif (
        "action_dist_inputs"
        in result
    ):

        action = result[
            "action_dist_inputs"
        ]

    else:

        raise RuntimeError(

            "Could not find an action in the "
            "RSAC RLModule output.\n"
            f"Available keys: {list(result.keys())}"

        )


    # --------------------------------------------------------
    # TORCH -> NUMPY
    # --------------------------------------------------------

    if isinstance(
        action,
        torch.Tensor
    ):

        action = (
            action.detach()
            .cpu()
            .numpy()
        )


    # --------------------------------------------------------
    # NUMPY
    # --------------------------------------------------------

    action = np.asarray(
        action,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # REMOVE BATCH DIMENSION
    # --------------------------------------------------------

    if action.ndim > 1:

        action = action[0]


    return action


# ============================================================
# HELPER:
# GET RSAC ACTION
# ============================================================

def get_rsac_action(
    agent,
    rsac_obs
):

    """
    RSAC inference using the NEW RLModule API.

    This replaces:

        sac.compute_single_action()

    because the SAC checkpoint uses RLlib's new API stack.
    """

    # --------------------------------------------------------
    # CONVERT OBSERVATION TO TORCH BATCH
    # --------------------------------------------------------

    obs_tensor = torch.from_numpy(
        rsac_obs
    ).float().unsqueeze(0)


    # --------------------------------------------------------
    # BUILD INPUT
    # --------------------------------------------------------

    input_dict = {

        Columns.OBS:
            obs_tensor

    }


    # --------------------------------------------------------
    # ADD RECURRENT STATE IF PRESENT
    # --------------------------------------------------------

    state_in = low_states[agent]


    if state_in is not None:

        if isinstance(
            state_in,
            (list, tuple)
        ):

            for i, state_value in enumerate(
                state_in
            ):

                if isinstance(
                    state_value,
                    np.ndarray
                ):

                    state_value = torch.from_numpy(
                        state_value
                    ).float()


                # Ensure batch dimension
                if isinstance(
                    state_value,
                    torch.Tensor
                ):

                    if state_value.ndim == 1:

                        state_value = (
                            state_value
                            .unsqueeze(0)
                        )


                input_dict[
                    f"state_in_{i}"
                ] = state_value


        elif isinstance(
            state_in,
            np.ndarray
        ):

            state_tensor = torch.from_numpy(
                state_in
            ).float()


            if state_tensor.ndim == 1:

                state_tensor = (
                    state_tensor
                    .unsqueeze(0)
                )


            input_dict[
                "state_in"
            ] = state_tensor


        # ----------------------------------------------------
        # SEQUENCE LENGTH
        # ----------------------------------------------------

        input_dict[
            Columns.SEQ_LENS
        ] = torch.ones(
            1,
            dtype=torch.int32
        )


    # --------------------------------------------------------
    # RL MODULE INFERENCE
    # --------------------------------------------------------

    result = sac_module.forward_inference(
        input_dict
    )


    # --------------------------------------------------------
    # UPDATE RECURRENT STATE IF RETURNED
    # --------------------------------------------------------

    if Columns.STATE_OUT in result:

        state_out = result[
            Columns.STATE_OUT
        ]

        if isinstance(
            state_out,
            (list, tuple)
        ):

            converted_state = []

            for x in state_out:

                if isinstance(
                    x,
                    torch.Tensor
                ):

                    x = (
                        x.detach()
                        .cpu()
                        .numpy()
                    )

                converted_state.append(
                    x
                )

            low_states[agent] = (
                converted_state
            )

        else:

            if isinstance(
                state_out,
                torch.Tensor
            ):

                state_out = (
                    state_out
                    .detach()
                    .cpu()
                    .numpy()
                )

            low_states[agent] = (
                state_out
            )


    # --------------------------------------------------------
    # EXTRACT ACTION
    # --------------------------------------------------------

    action = extract_sac_action(
        result
    )


    # --------------------------------------------------------
    # ENSURE 3-D ACTION
    # --------------------------------------------------------

    if action.size < 3:

        padded = np.zeros(
            3,
            dtype=np.float32
        )

        padded[
            :action.size
        ] = action

        action = padded

    else:

        action = action[:3]


    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------

    action = np.clip(
        action,
        -1.0,
        1.0
    )


    return action.astype(
        np.float32
    )


# ============================================================
# EVALUATION SETTINGS
# ============================================================

MAX_STEPS = 50

MACRO_INTERVAL = 10


# ============================================================
# INITIAL METRICS
# ============================================================

total_reward = 0.0

total_collisions = 0

total_path_length = 0.0

mission_success = False


# ============================================================
# STEP LOOP
# ============================================================

print("\n======================================")
print("STARTING ABLATION EVALUATION")
print("WITHOUT MID-LEVEL")
print("======================================\n")


for t in range(
    MAX_STEPS
):


    # ========================================================
    # 1. MAPPO STRATEGIC DECISION
    # ========================================================

    if t % MACRO_INTERVAL == 0:

        for agent in agents:

            get_mappo_action(
                agent,
                obs[agent]
            )


    # ========================================================
    # 2. DIRECT MAPPO -> RSAC
    # ========================================================

    actions = {}


    for agent in agents:


        # ----------------------------------------------------
        # BUILD RSAC OBSERVATION
        # ----------------------------------------------------

        rsac_obs = build_rsac_observation(

            agent,

            macro_actions[agent]

        )


        # ----------------------------------------------------
        # RSAC ACTION
        # ----------------------------------------------------

        action = get_rsac_action(

            agent,

            rsac_obs

        )


        # ----------------------------------------------------
        # STORE ACTION
        # ----------------------------------------------------

        actions[agent] = action


    # ========================================================
    # 3. ENVIRONMENT STEP
    # ========================================================

    next_obs, rewards, terms, truncs, infos = (
        env.step(actions)
    )


    # ========================================================
    # 4. STEP REWARD
    # ========================================================

    step_reward = 0.0


    for agent in agents:

        step_reward += float(

            rewards.get(
                agent,
                0.0
            )

        )


    total_reward += step_reward


    episode_rewards.append(
        total_reward
    )


    # ========================================================
    # 5. UPDATE PREVIOUS MAPPO REWARDS
    # ========================================================

    for agent in agents:

        prev_macro_rewards[agent] = float(

            rewards.get(
                agent,
                0.0
            )

        )


    # ========================================================
    # 6. TRAJECTORY
    # ========================================================

    for agent in agents:

        current_position = np.asarray(

            env.pos[agent],

            dtype=np.float32

        )


        trajectories[agent].append(

            current_position.copy()

        )


        # ----------------------------------------------------
        # COVERAGE CELL
        # ----------------------------------------------------

        cell = tuple(

            np.floor(
                current_position
            ).astype(int)

        )


        visited_cells.add(
            cell
        )


    # ========================================================
    # 7. PATH LENGTH
    # ========================================================

    step_path = 0.0


    for agent in agents:

        trajectory = trajectories[agent]


        if len(trajectory) >= 2:

            displacement = (

                trajectory[-1]

                -

                trajectory[-2]

            )


            step_path += np.linalg.norm(
                displacement
            )


    total_path_length += step_path


    path_length_history.append(
        total_path_length
    )


    # ========================================================
    # 8. UAV-UAV COLLISIONS
    # ========================================================

    step_collisions = 0


    for i in range(
        len(agents)
    ):

        for j in range(
            i + 1,
            len(agents)
        ):

            a1 = agents[i]

            a2 = agents[j]


            distance = np.linalg.norm(

                env.pos[a1]

                -

                env.pos[a2]

            )


            if distance < env.safe_distance:

                step_collisions += 1


    total_collisions += (
        step_collisions
    )


    collision_history.append(
        step_collisions
    )


    # ========================================================
    # 9. TASK CONFLICT
    # ========================================================

    assigned_tasks = [

        env.task_assignments[agent]

        for agent in agents

    ]


    conflicts = (

        len(assigned_tasks)

        -

        len(
            set(assigned_tasks)
        )

    )


    task_conflict_history.append(
        conflicts
    )


    # ========================================================
    # 10. COVERAGE
    # ========================================================

    coverage_history.append(

        len(
            visited_cells
        )

    )


    # ========================================================
    # 11. MISSION SUCCESS
    # ========================================================

    mission_success = bool(

        terms.get(
            "__all__",
            False
        )

    )


    success_history.append(

        int(
            mission_success
        )

    )


    # ========================================================
    # 12. UPDATE OBSERVATION
    # ========================================================

    obs = next_obs


    # ========================================================
    # 13. PRINT PROGRESS
    # ========================================================

    if t % 10 == 0:

        print(

            f"Step {t:3d} | "

            f"Reward = "
            f"{total_reward:9.3f} | "

            f"Collisions = "
            f"{total_collisions:3d} | "

            f"Task conflicts = "
            f"{conflicts:2d} | "

            f"Coverage = "
            f"{len(visited_cells):4d}"

        )


    # ========================================================
    # 14. TERMINATION
    # ========================================================

    if mission_success:

        print(
            "\nMission completed."
        )

        break


    if bool(

        truncs.get(
            "__all__",
            False
        )

    ):

        print(
            "\nMaximum episode length reached."
        )

        break


# ============================================================
# FINAL METRICS
# ============================================================

steps_completed = len(
    collision_history
)


average_reward = (

    total_reward

    /

    max(
        steps_completed,
        1
    )

)


average_collisions = (

    total_collisions

    /

    max(
        steps_completed,
        1
    )

)


average_task_conflicts = (

    np.mean(
        task_conflict_history
    )

    if task_conflict_history

    else 0.0

)


final_coverage = len(
    visited_cells
)


success_rate = 100.0 if mission_success else 0.0

#===========================================
# PRINT FINAL RESULTS
# ============================================================

print("\n")

print(
    "===================================================="
)

print(
    " WITHOUT MID-LEVEL — FINAL RESULTS"
)

print(
    "===================================================="
)


print(
    f"Steps completed          : "
    f"{steps_completed}"
)


print(
    f"Total reward             : "
    f"{total_reward:.4f}"
)


print(
    f"Average step reward      : "
    f"{average_reward:.4f}"
)


print(
    f"Total UAV collisions     : "
    f"{total_collisions}"
)


print(
    f"Average collisions/step  : "
    f"{average_collisions:.6f}"
)


print(
    f"Average task conflicts   : "
    f"{average_task_conflicts:.4f}"
)


print(
    f"Coverage cells           : "
    f"{final_coverage}"
)


print(
    f"Total path length        : "
    f"{total_path_length:.4f}"
)


print(
    f"Mission success          : "
    f"{mission_success}"
)


print(
    f"Success rate             : "
    f"{success_rate:.2f}%"
)


print(
    "===================================================="
)


# ============================================================
# SAVE RESULTS
# ============================================================

np.savez(

    RESULT_FILE,

    total_reward=total_reward,

    average_reward=average_reward,

    total_collisions=total_collisions,

    average_collisions=average_collisions,

    average_task_conflicts=average_task_conflicts,

    coverage=final_coverage,

    total_path_length=total_path_length,

    success_rate=success_rate,

    episode_rewards=np.asarray(
        episode_rewards
    ),

    collision_history=np.asarray(
        collision_history
    ),

    coverage_history=np.asarray(
        coverage_history
    ),

    success_history=np.asarray(
        success_history
    ),

    path_length_history=np.asarray(
        path_length_history
    ),

    task_conflict_history=np.asarray(
        task_conflict_history
    )

)


print(
    "\nResults saved to:"
)

print(
    RESULT_FILE
)


# ============================================================
# PLOT 1 — CUMULATIVE REWARD
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    episode_rewards
)


plt.xlabel(
    "Environment Step"
)


plt.ylabel(
    "Cumulative Reward"
)


plt.title(
    "MA-HMARL Without Mid-Level: Reward"
)


plt.grid()


plt.savefig(
    REWARD_PLOT,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# PLOT 2 — COLLISIONS
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    collision_history
)


plt.xlabel(
    "Environment Step"
)


plt.ylabel(
    "UAV-UAV Collision Count"
)


plt.title(
    "MA-HMARL Without Mid-Level: Collisions"
)


plt.grid()


plt.savefig(
    COLLISION_PLOT,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# PLOT 3 — COVERAGE
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    coverage_history
)


plt.xlabel(
    "Environment Step"
)


plt.ylabel(
    "Visited Grid Cells"
)


plt.title(
    "MA-HMARL Without Mid-Level: Coverage"
)


plt.grid()


plt.savefig(
    COVERAGE_PLOT,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# SHUTDOWN
# ============================================================

#ray.shutdown()


print(
    "\nRay shutdown completed."
)

print(
    "Ablation evaluation completed successfully."
)