from pettingzoo.utils.env import ParallelEnv
import numpy as np
from gymnasium import spaces


class UAVHMARLEnv(ParallelEnv):
    metadata = {"name": "uav_hmarl_v6"}

    def __init__(self, num_uavs=5, dropout_prob=0.3):
        self.num_uavs = num_uavs
        self.agents = [f"uav_{i}" for i in range(num_uavs)]

        self.dropout_prob = dropout_prob

        self.r_comm = 6.0
        self.max_steps = 100   # 🔥 shorter episodes
        self.num_tasks = 5

        # ✅ normalized obs
        self.observation_spaces = {
            a: spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
            for a in self.agents
        }

        self.action_spaces = {
            a: spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
            for a in self.agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ========================
    # RESET
    # ========================
    def reset(self, *, seed=None, options=None):
        self.step_count = 0

        self.pos = {a: np.random.uniform(0, 5, 3) for a in self.agents}
        self.vel = {a: np.zeros(3) for a in self.agents}

        self.tasks = np.random.uniform(0, 5, (self.num_tasks, 3))

        self.task_assignments = {
            a: i % self.num_tasks for i, a in enumerate(self.agents)
        }

        self.prev_dist = None

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

        # 🔥 dropout communication
        if neighbors and np.random.rand() > self.dropout_prob:
            msg = np.mean([self.pos[n] for n in neighbors], axis=0)

        task = self.tasks[self.task_assignments[agent]]

        obs = np.concatenate([
            self.pos[agent],
            self.vel[agent],
            task,
            msg
        ]).astype(np.float32)

        # ✅ better normalization
        obs[:3] /= 20.0      # position
        obs[3:6] /= 1.0      # velocity (already small)
        obs[6:9] /= 20.0     # task
        obs[9:12] /= 20.0    # comm

        obs = np.clip(obs, -1.0, 1.0)
        return obs

    # ========================
    # STEP
    # ========================
    def step(self, actions):
        self.step_count += 1

        rewards, terms, truncs, infos = {}, {}, {}, {}

        # ===== movement =====
        for a, act in actions.items():
            move = np.clip(act, -1, 1)
            move = 0.3 * move

            self.vel[a] = move
            self.pos[a] += move

            # 🔥 prevent explosion
            self.pos[a] = np.clip(self.pos[a], -20.0, 20.0)

        # ===== distance calc =====
        positions = np.array(list(self.pos.values()))
        task_positions = np.array([
            self.tasks[self.task_assignments[a]]
            for a in self.agents
        ])

        dist = np.linalg.norm(positions - task_positions, axis=1)
        mean_dist = np.mean(dist)

        # ===== REWARD FIX =====
        # smooth progress (NO normalization!)
        if self.prev_dist is None:
            progress = 0.0
        else:
            progress = self.prev_dist - mean_dist

        self.prev_dist = mean_dist

        R_progress = 5.0 * progress           # 🔥 stable signal
        R_distance = -0.01 * mean_dist        # 🔥 small penalty

        success = np.any(dist < 0.5)
        R_success = 50.0 if success else 0.0  # 🔥 strong signal

        final_reward = R_progress + R_distance + R_success

        timeout = self.step_count >= self.max_steps

        for a in self.agents:
            rewards[a] = float(final_reward)
            terms[a] = success
            truncs[a] = timeout
            infos[a] = {}

        terms["__all__"] = success
        truncs["__all__"] = timeout

        obs = {a: self._obs(a) for a in self.agents}

        return obs, rewards, terms, truncs, infos