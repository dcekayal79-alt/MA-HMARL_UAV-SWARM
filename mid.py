# mid_level_task.py

import numpy as np


class MidLevelPlanner:
    """
    Mid-level planner for HMARL:
    - Takes macro (leader) output
    - Converts it into task assignments
    - Produces:
        1) Final task per agent
        2) Intent vector [dx, dy, dz, distance, reached_flag]
    """

    def __init__(self, refine=True, bias_strength=1.0):
        self.refine = refine
        self.bias_strength = bias_strength

    # ========================
    # Score function φ(o_i, z_j)
    # ========================
    def score(self, obs, task):
        return -np.linalg.norm(obs[:3] - task)

    # ========================
    # SAFE MACRO → TASK INDEX
    # ========================
    def macro_to_task_index(self, macro_action, tasks):
        """
        Converts macro output into valid task index
        Handles:
            - discrete index
            - continuous vector → nearest task
        """

        # Case 1: integer index
        if isinstance(macro_action, (int, np.integer)):
            return int(np.clip(macro_action, 0, len(tasks) - 1))

        # Case 2: vector → nearest task
        macro_action = np.array(macro_action)
        distances = np.linalg.norm(tasks - macro_action[:3], axis=1)
        return int(np.argmin(distances))

    # ========================
    # TASK ASSIGNMENT (WITH COLLISION AVOIDANCE)
    # ========================
    def assign_tasks(self, observations, tasks, macro_actions):
        """
        Prevents multiple agents picking same task
        """

        assignments = {}
        used_tasks = set()

        for agent, obs in observations.items():

            macro_action = macro_actions[agent]
            leader_task_idx = self.macro_to_task_index(macro_action, tasks)

            if not self.refine:
                assignments[agent] = tasks[leader_task_idx]
                used_tasks.add(leader_task_idx)
                continue

            # Score all tasks
            scores = np.array([
                self.score(obs, t) for t in tasks
            ])

            # Bias toward macro-selected task
            scores[leader_task_idx] += self.bias_strength

            # 🔴 Penalize already used tasks
            for t in used_tasks:
                scores[t] -= 1000  # large penalty

            best_idx = int(np.argmax(scores))

            assignments[agent] = tasks[best_idx]
            used_tasks.add(best_idx)

        return assignments

    # ========================
    # INTENT GENERATION (FIXED)
    # ========================
    def compute_intent(self, obs, task):
        """
        Returns:
        [dx, dy, dz, distance, reached_flag]
        """

        direction = task - obs[:3]
        distance = np.linalg.norm(direction)

        # 🔴 FIX: handle "already at task"
        if distance < 1e-3:
            unit_dir = np.zeros_like(direction)
            reached = 1.0
            distance = 0.0
        else:
            unit_dir = direction / distance
            reached = 0.0

        intent = np.concatenate([unit_dir, [distance, reached]])

        return intent

    # ========================
    # FULL PIPELINE
    # ========================
    def plan(self, observations, tasks, macro_actions):
        """
        returns:
            assignments: {agent: task_position}
            intents: {agent: intent_vector}
        """

        assignments = self.assign_tasks(observations, tasks, macro_actions)

        intents = {
            agent: self.compute_intent(observations[agent], task)
            for agent, task in assignments.items()
        }

        return assignments, intents


# ========================
# TEST BLOCK
# ========================
if __name__ == "__main__":

    planner = MidLevelPlanner()

    # Dummy data
    observations = {
        "uav_0": np.array([0.0, 0.0, 0.0]),
        "uav_1": np.array([5.0, 5.0, 0.0]),
    }

    tasks = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [5.0, 5.0, 0.0],
    ])

    # Macro outputs (mixed types)
    macro_actions = {
        "uav_0": 0,                          # index
        "uav_1": np.array([4.5, 5.2, 0.0]), # vector
    }

    assignments, intents = planner.plan(observations, tasks, macro_actions)

    print("Assignments:")
    for k, v in assignments.items():
        print(k, "->", v)

    print("\nIntents:")
    for k, v in intents.items():
        print(k, "->", v)