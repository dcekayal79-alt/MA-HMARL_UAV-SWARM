from pettingzoo.utils.env import ParallelEnv
import numpy as np
from gymnasium import spaces


class UAVHMARLEnv(ParallelEnv):

    metadata = {"name": "memory_aware_hmarl_env_v1"}

    # =========================================================
    # INIT
    # =========================================================
    def __init__(

        self,
        num_uavs=5,
        num_tasks=5,
        curriculum_level=0

    ):

        super().__init__()

        self.num_uavs = num_uavs
        self.num_tasks = num_tasks

        self.agents = [
            f"uav_{i}"
            for i in range(num_uavs)
        ]

        self.possible_agents = self.agents[:]

        # =====================================================
        # WORLD SETTINGS
        # =====================================================
        self.world_size = 20.0

        self.max_steps = 200

        self.r_comm = 6.0

        self.safe_distance = 0.5

        # =====================================================
        # CURRICULUM SETTINGS
        # =====================================================
        self.curriculum_level = curriculum_level

        self._set_curriculum()

        # =====================================================
        # OBSERVATION SPACE
        #
        # pos(3)
        # vel(3)
        # task(3)
        # msg(3)
        # closest_neighbor_dist(1)
        #
        # TOTAL = 13
        # =====================================================
        self.observation_spaces = {

            agent: spaces.Box(

                low=-1.0,
                high=1.0,
                shape=(13,),
                dtype=np.float32

            )

            for agent in self.agents
        }

        # =====================================================
        # ACTION SPACE
        #
        # vx, vy, vz
        # =====================================================
        self.action_spaces = {

            agent: spaces.Box(

                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32

            )

            for agent in self.agents
        }

    # =========================================================
    # CURRICULUM
    # =========================================================
    def _set_curriculum(self):

        if self.curriculum_level == 0:

            self.dropout_prob = 0.1
            self.r_comm = 8.0

        elif self.curriculum_level == 1:

            self.dropout_prob = 0.3
            self.r_comm = 7.0

        elif self.curriculum_level == 2:

            self.dropout_prob = 0.5
            self.r_comm = 6.0

        else:

            self.dropout_prob = 0.7
            self.r_comm = 5.0

    # =========================================================
    # SPACES
    # =========================================================
    def observation_space(self, agent):

        return self.observation_spaces[agent]

    def action_space(self, agent):

        return self.action_spaces[agent]

    # =========================================================
    # RESET
    # =========================================================
    def reset(self, *, seed=None, options=None):

        self.agents = self.possible_agents[:]

        self.step_count = 0

        # =====================================================
        # UAV POSITIONS
        # =====================================================
        self.pos = {

            agent: np.random.uniform(
                0,
                5,
                size=3
            ).astype(np.float32)

            for agent in self.agents
        }

        # =====================================================
        # UAV VELOCITIES
        # =====================================================
        self.vel = {

            agent: np.zeros(
                3,
                dtype=np.float32
            )

            for agent in self.agents
        }

        # =====================================================
        # TASKS
        # =====================================================
        self.tasks = np.random.uniform(

            0,
            5,
            size=(self.num_tasks, 3)

        ).astype(np.float32)

        # =====================================================
        # TASK ASSIGNMENTS
        # =====================================================
        self.task_assignments = {

            agent: i % self.num_tasks

            for i, agent in enumerate(self.agents)
        }

        # =====================================================
        # TRUE PARTIAL OBSERVABILITY
        #
        # persistent hidden task
        # =====================================================
        self.hidden_mask = {

            agent: np.random.rand() < (
                0.2 + 0.15 * self.curriculum_level
            )

            for agent in self.agents
        }

        # =====================================================
        # METRICS
        # =====================================================
        self.prev_dist = None

        self.collision_count = 0

        self.success_count = 0

        self.communication_success = 0

        # =====================================================
        # TRAJECTORY LOGGING
        # =====================================================
        self.trajectory_log = {

            agent: []

            for agent in self.agents
        }

        observations = {

            agent: self._obs(agent)

            for agent in self.agents
        }

        infos = {

            agent: {}

            for agent in self.agents
        }

        return observations, infos

    # =========================================================
    # NEIGHBORS
    # =========================================================
    def _neighbors(self, agent):

        neighbors = []

        for other in self.agents:

            if other == agent:
                continue

            dist = np.linalg.norm(

                self.pos[agent] -
                self.pos[other]

            )

            if dist <= self.r_comm:

                neighbors.append(other)

        return neighbors

    # =========================================================
    # OBSERVATION
    # =========================================================
    def _obs(self, agent):

        neighbors = self._neighbors(agent)

        # =====================================================
        # COMMUNICATION MESSAGE
        # =====================================================
        msg = np.zeros(3, dtype=np.float32)

        if (

            len(neighbors) > 0 and
            np.random.rand() > self.dropout_prob

        ):

            msg = np.mean(

                [self.pos[n] for n in neighbors],
                axis=0

            ).astype(np.float32)

            self.communication_success += 1

        # =====================================================
        # PARTIAL OBSERVABILITY
        # =====================================================
        if self.hidden_mask[agent]:

            task = np.zeros(
                3,
                dtype=np.float32
            )

        else:

            task = self.tasks[
                self.task_assignments[agent]
            ]

        # =====================================================
        # CLOSEST NEIGHBOR DIST
        # =====================================================
        closest_neighbor_dist = np.array(
            [1.0],
            dtype=np.float32
        )

        if len(neighbors) > 0:

            dists = [

                np.linalg.norm(
                    self.pos[agent] -
                    self.pos[n]
                )

                for n in neighbors
            ]

            closest_neighbor_dist[0] = (

                min(dists) / self.r_comm

            )

        # =====================================================
        # BUILD OBS
        # =====================================================
        obs = np.concatenate([

            self.pos[agent],          # 3
            self.vel[agent],          # 3
            task,                     # 3
            msg,                      # 3
            closest_neighbor_dist     # 1

        ]).astype(np.float32)

        # =====================================================
        # NORMALIZATION
        # =====================================================
        obs[0:3] /= self.world_size

        obs[3:6] /= 1.0

        obs[6:9] /= self.world_size

        obs[9:12] /= self.world_size

        obs = np.clip(
            obs,
            -1.0,
            1.0
        )

        return obs.astype(np.float32)

    # =========================================================
    # STEP
    # =========================================================
    def step(self, actions):

        self.step_count += 1

        rewards = {}

        terminations = {}

        truncations = {}

        infos = {}

        # =====================================================
        # UAV MOVEMENT
        # =====================================================
        for agent, action in actions.items():

            move = np.clip(
                action,
                -1.0,
                1.0
            )

            move = 0.2 * move

            self.vel[agent] = move.astype(
                np.float32
            )

            self.pos[agent] += self.vel[agent]

            # world boundary
            self.pos[agent] = np.clip(

                self.pos[agent],
                -self.world_size,
                self.world_size

            )

            # =================================================
            # TRAJECTORY LOGGING
            # =================================================
            self.trajectory_log[agent].append(
                self.pos[agent].copy()
            )

        # =====================================================
        # TASK DISTANCES
        # =====================================================
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

        # =====================================================
        # PROGRESS REWARD
        # =====================================================
        if self.prev_dist is None:

            progress = 0.0

        else:

            progress = (

                self.prev_dist -
                mean_dist

            )

        self.prev_dist = mean_dist

        R_progress = 10.0 * progress

        R_distance = -0.05 * mean_dist

        # =====================================================
        # SUCCESS BONUS
        # =====================================================
        success = np.any(

            np.array(distances) < 0.5

        )

        R_success = (

            100.0 if success else 0.0

        )

        if success:

            self.success_count += 1

        # =====================================================
        # COLLISION PENALTY
        # =====================================================
        collision_penalty = 0.0

        for i, a1 in enumerate(self.agents):

            for j, a2 in enumerate(self.agents):

                if i >= j:
                    continue

                dist = np.linalg.norm(

                    self.pos[a1] -
                    self.pos[a2]

                )

                if dist < self.safe_distance:

                    collision_penalty -= 10.0

                    self.collision_count += 1

        # =====================================================
        # FINAL REWARD
        # =====================================================
        final_reward = (

            R_progress +
            R_distance +
            R_success +
            collision_penalty

        )

        timeout = (

            self.step_count >= self.max_steps

        )

        # =====================================================
        # MULTI-AGENT OUTPUTS
        # =====================================================
        for agent in self.agents:

            rewards[agent] = float(
                final_reward
            )

            terminations[agent] = success

            truncations[agent] = timeout

            infos[agent] = {

                "mean_distance": mean_dist,
                "collisions": self.collision_count,
                "successes": self.success_count,
                "communications": self.communication_success
            }

        terminations["__all__"] = success

        truncations["__all__"] = timeout

        observations = {

            agent: self._obs(agent)

            for agent in self.agents
        }

        return (

            observations,
            rewards,
            terminations,
            truncations,
            infos

        )