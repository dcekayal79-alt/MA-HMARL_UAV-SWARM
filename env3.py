from pettingzoo.utils.env import ParallelEnv
import numpy as np
from gymnasium import spaces


class UAVHMARLEnv(ParallelEnv):
    metadata = {"name": "uav_hmarl_v3"}

    def __init__(self, num_uavs=5):
        self.num_uavs = num_uavs
        self.agents = [f"uav_{i}" for i in range(num_uavs)]

        self.leader = self.agents[0]
        self.followers = self.agents[1:]

        self.r_comm = 6.0
        self.d_safe = 1.0
        self.max_steps = 200

        self.num_tasks = 5

        # Observation: pos(3) + vel(3) + task(3) + neighbor msg(3)
        self.observation_spaces = {
            a: spaces.Box(-20, 20, shape=(12,), dtype=np.float32)
            for a in self.agents
        }

        # Action space (simplified)
        self.action_spaces = {
            a: spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
            for a in self.agents
        }

        self.prev_dist = None
        self.reset()

    # ========================
    # SPACES
    # ========================
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ========================
    # RESET
    # ========================
    def reset(self, *, seed=None, options=None):
        self.step_count = 0

        # Easier start (important)
        self.pos = {a: np.random.uniform(0, 5, 3) for a in self.agents}
        self.vel = {a: np.zeros(3) for a in self.agents}

        self.tasks = np.random.uniform(0, 5, (self.num_tasks, 3))

        self.task_assignments = {
            a: i % self.num_tasks for i, a in enumerate(self.agents)
        }

        # ✅ initialize prev_dist (CRITICAL FIX)
        positions = np.array(list(self.pos.values()))
        task_positions = np.array([
            self.tasks[self.task_assignments[a]]
            for a in self.agents
        ])

        self.prev_dist = np.linalg.norm(positions - task_positions, axis=1)

        obs = {a: self._obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}

        return obs, infos

    # ========================
    # OBS
    # ========================
    def _neighbors(self, agent):
        return [
            other for other in self.agents
            if other != agent and
            np.linalg.norm(self.pos[agent] - self.pos[other]) <= self.r_comm
        ]

    def _obs(self, agent):
        neighbors = self._neighbors(agent)

        msg = np.zeros(3)
        if neighbors:
            msg = np.mean([self.pos[n] for n in neighbors], axis=0)

        task = self.tasks[self.task_assignments[agent]]

        return np.concatenate([
            self.pos[agent],
            self.vel[agent],
            task,
            msg
        ]).astype(np.float32)

    # ========================
    # STEP
    # ========================
    def step(self, actions):
        self.step_count += 1

        rewards, terms, truncs, infos = {}, {}, {}, {}

        # APPLY ACTIONS
        for a, act in actions.items():
            move = np.clip(act, -1, 1)
            move = 0.3 * move  # stability

            self.vel[a] = move
            self.pos[a] += move

                # ========================
        # REWARD (FIXED + STRONG SIGNAL)
        # ========================
        positions = np.array(list(self.pos.values()))
        task_positions = np.array([
            self.tasks[self.task_assignments[a]]
            for a in self.agents
        ])

        dist = np.linalg.norm(positions - task_positions, axis=1)

        # ------------------------
        # PROGRESS (MAIN SIGNAL)
        # ------------------------
        if self.prev_dist is None:
            progress = np.zeros_like(dist)
        else:
            progress = self.prev_dist - dist

        R_progress = np.mean(progress)

        # amplify progress (IMPORTANT)
        R_progress *= 10.0

        # ------------------------
        # DISTANCE SHAPING
        # ------------------------
        R_distance = -0.05 * np.mean(dist)

        # ------------------------
        # SUCCESS BONUS (EASIER)
        # ------------------------
        success_bonus = 5.0 if np.any(dist < 1.0) else 0.0

        # ------------------------
        # COLLISION PENALTY
        # ------------------------
        R_collision = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if np.linalg.norm(positions[i] - positions[j]) < self.d_safe:
                    R_collision -= 0.1

        # ------------------------
        # FINAL REWARD
        # ------------------------
        final_reward = (
            R_progress +
            R_distance +
            success_bonus +
            R_collision
        )

        final_reward /= self.num_uavs

        # update memory
        self.prev_dist = dist.copy()

        # ========================
        # TERMINATION
        # ========================
        success = np.any(dist < 0.5)
        timeout = self.step_count >= self.max_steps

        for a in self.agents:
            rewards[a] = final_reward
            terms[a] = success
            truncs[a] = timeout
            infos[a] = {}

        terms["__all__"] = success
        truncs["__all__"] = timeout

        obs = {a: self._obs(a) for a in self.agents}

        return obs, rewards, terms, truncs, infos

    # ========================
    # RENDER
    # ========================
    def render(self):
        import matplotlib.pyplot as plt

        positions = np.array(list(self.pos.values()))

        plt.clf()
        plt.scatter(positions[:, 0], positions[:, 1], label="UAVs")

        for i, task in enumerate(self.tasks):
            plt.scatter(task[0], task[1], marker='x')
            plt.text(task[0], task[1], f"T{i}")

        plt.title(f"Step {self.step_count}")
        plt.legend()
        plt.pause(0.01)