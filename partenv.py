from pettingzoo.utils.env import ParallelEnv
import numpy as np
from gymnasium import spaces


class UAVHMARLEnv(ParallelEnv):
    metadata = {"name": "uav_hmarl_v7"}

    def __init__(self, num_uavs=5, dropout_prob=0.7):
        super().__init__()

        self.num_uavs = num_uavs
        self.agents = [f"uav_{i}" for i in range(num_uavs)]

        self.dropout_prob = dropout_prob

        self.r_comm = 6.0
        self.max_steps = 100
        self.num_tasks = 5

        # ====================================
        # OBS/ACTION SPACES
        # ====================================
        self.observation_spaces = {
            a: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(12,),
                dtype=np.float32
            )
            for a in self.agents
        }

        self.action_spaces = {
            a: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32
            )
            for a in self.agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ====================================
    # RESET
    # ====================================
    def reset(self, *, seed=None, options=None):

        self.step_count = 0

        self.pos = {
            a: np.random.uniform(0, 5, 3).astype(np.float32)
            for a in self.agents
        }

        self.vel = {
            a: np.zeros(3, dtype=np.float32)
            for a in self.agents
        }

        self.tasks = np.random.uniform(
            0,
            5,
            (self.num_tasks, 3)
        ).astype(np.float32)

        self.task_assignments = {
            a: i % self.num_tasks
            for i, a in enumerate(self.agents)
        }

        # ====================================
        # TRUE PARTIAL OBSERVABILITY
        # Persistent hidden task mask
        # ====================================
        self.hidden_mask = {
            a: np.random.rand() < 0.7
            for a in self.agents
        }

        self.prev_dist = None

        obs = {
            a: self._obs(a)
            for a in self.agents
        }

        infos = {
            a: {}
            for a in self.agents
        }

        return obs, infos

    # ====================================
    # NEIGHBORS
    # ====================================
    def _neighbors(self, agent):

        return [
            other for other in self.agents
            if other != agent and
            np.linalg.norm(
                self.pos[agent] - self.pos[other]
            ) <= self.r_comm
        ]

    # ====================================
    # OBSERVATION
    # ====================================
    def _obs(self, agent):

        neighbors = self._neighbors(agent)

        # ====================================
        # COMMUNICATION MESSAGE
        # ====================================
        msg = np.zeros(3, dtype=np.float32)

        if (
            len(neighbors) > 0 and
            np.random.rand() > self.dropout_prob
        ):
            msg = np.mean(
                [self.pos[n] for n in neighbors],
                axis=0
            ).astype(np.float32)

        # ====================================
        # PERSISTENT TASK MASKING
        # ====================================
        if self.hidden_mask[agent]:
            task = np.zeros(3, dtype=np.float32)
        else:
            task = self.tasks[
                self.task_assignments[agent]
            ]

        # ====================================
        # BUILD OBS
        # ALWAYS 12 DIMENSIONS
        # ====================================
        obs = np.concatenate([
            self.pos[agent],   # 3
            self.vel[agent],   # 3
            task,              # 3
            msg                # 3
        ]).astype(np.float32)

        # ====================================
        # NORMALIZATION
        # ====================================
        obs[0:3] /= 20.0
        obs[3:6] /= 1.0
        obs[6:9] /= 20.0
        obs[9:12] /= 20.0

        obs = np.clip(obs, -1.0, 1.0)

        return obs.astype(np.float32)

    # ====================================
    # STEP
    # ====================================
    def step(self, actions):

        self.step_count += 1

        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # ====================================
        # UAV MOVEMENT
        # ====================================
        for agent, action in actions.items():

            move = np.clip(action, -1.0, 1.0)
            move = 0.2 * move

            self.vel[agent] = move.astype(np.float32)

            self.pos[agent] += self.vel[agent]

            self.pos[agent] = np.clip(
                self.pos[agent],
                -20.0,
                20.0
            )

        # ====================================
        # DISTANCE TO TASKS
        # ====================================
        distances = []

        for agent in self.agents:

            task = self.tasks[
                self.task_assignments[agent]
            ]

            dist = np.linalg.norm(
                self.pos[agent] - task
            )

            distances.append(dist)

        mean_dist = np.mean(distances)

        # ====================================
        # PROGRESS REWARD
        # ====================================
        if self.prev_dist is None:
            progress = 0.0
        else:
            progress = self.prev_dist - mean_dist

        self.prev_dist = mean_dist

        # ====================================
        # REWARD FUNCTION
        # ====================================
        R_progress = 10.0 * progress
        R_distance = -0.05 * mean_dist

        success = np.any(
            np.array(distances) < 0.5
        )

        R_success = 100.0 if success else 0.0

        final_reward = (
            R_progress +
            R_distance +
            R_success
        )

        timeout = (
            self.step_count >= self.max_steps
        )

        # ====================================
        # MULTI-AGENT OUTPUTS
        # ====================================
        for agent in self.agents:

            rewards[agent] = float(final_reward)

            terminations[agent] = success

            truncations[agent] = timeout

            infos[agent] = {}

        terminations["__all__"] = success
        truncations["__all__"] = timeout

        observations = {
            a: self._obs(a)
            for a in self.agents
        }

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos
        )