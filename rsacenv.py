import gymnasium as gym
import numpy as np

from gymnasium import spaces


class UAVRSACEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self):

        super().__init__()

        # ==========================================
        # ENV SETTINGS
        # ==========================================
        self.world_size = 20.0
        self.max_steps = 100

        self.safe_distance = 0.5
        self.sensor_range = 6.0

        # single UAV
        self.num_obstacles = 3

        # ==========================================
        # OBSERVATION
        #
        # position(3)
        # velocity(3)
        # target(3)
        # nearest_obstacle_vector(3)
        # nearest_obstacle_distance(1)
        #
        # TOTAL = 13
        # ==========================================
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(13,),
            dtype=np.float32
        )

        # ==========================================
        # ACTION SPACE
        #
        # vx vy vz
        # ==========================================
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

    # ==================================================
    # RESET
    # ==================================================
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.step_count = 0

        # UAV state
        self.pos = np.random.uniform(
            0,
            5,
            size=3
        ).astype(np.float32)

        self.vel = np.zeros(
            3,
            dtype=np.float32
        )

        # target
        self.target = np.random.uniform(
            0,
            5,
            size=3
        ).astype(np.float32)

        # obstacles
        self.obstacles = np.random.uniform(
            0,
            10,
            size=(self.num_obstacles, 3)
        ).astype(np.float32)

        # ==========================================
        # PARTIAL OBSERVABILITY
        #
        # sometimes hide target
        # ==========================================
        self.target_hidden = (
            np.random.rand() < 0.5
        )

        self.prev_dist = np.linalg.norm(
            self.pos - self.target
        )

        obs = self._get_obs()

        return obs, {}

    # ==================================================
    # OBSERVATION
    # ==================================================
    def _get_obs(self):

        # ==========================================
        # TARGET OBS
        # ==========================================
        if self.target_hidden:

            target_obs = np.zeros(
                3,
                dtype=np.float32
            )

        else:

            target_obs = self.target.copy()

        # ==========================================
        # LOCAL OBSTACLE SENSOR
        # ==========================================
        nearest_vec = np.zeros(
            3,
            dtype=np.float32
        )

        nearest_dist = 1.0

        dists = np.linalg.norm(
            self.obstacles - self.pos,
            axis=1
        )

        idx = np.argmin(dists)

        if dists[idx] <= self.sensor_range:

            nearest_vec = (
                self.obstacles[idx] - self.pos
            )

            nearest_dist = (
                dists[idx] / self.sensor_range
            )

        # ==========================================
        # BUILD OBS
        # ==========================================
        obs = np.concatenate([

            self.pos,
            self.vel,
            target_obs,
            nearest_vec,
            np.array([nearest_dist])

        ]).astype(np.float32)

        # normalize
        obs[0:3] /= self.world_size
        obs[3:6] /= 1.0
        obs[6:9] /= self.world_size
        obs[9:12] /= self.sensor_range

        obs = np.clip(
            obs,
            -1.0,
            1.0
        )

        return obs.astype(np.float32)

    # ==================================================
    # STEP
    # ==================================================
    def step(self, action):

        self.step_count += 1

        # ==========================================
        # UAV MOTION
        # ==========================================
        move = np.clip(
            action,
            -1.0,
            1.0
        )

        move = 0.2 * move

        self.vel = move.astype(np.float32)

        self.pos += self.vel

        self.pos = np.clip(
            self.pos,
            -self.world_size,
            self.world_size
        )

        # ==========================================
        # DISTANCE TO TARGET
        # ==========================================
        dist = np.linalg.norm(
            self.pos - self.target
        )

        progress = (
            self.prev_dist - dist
        )

        self.prev_dist = dist

        # ==========================================
        # REWARDS
        # ==========================================
        R_progress = 10.0 * progress

        R_distance = -0.05 * dist

        success = dist < 0.5

        R_success = (
            100.0 if success else 0.0
        )

        # ==========================================
        # COLLISION PENALTY
        # ==========================================
        collision = False

        collision_penalty = 0.0

        obstacle_dists = np.linalg.norm(
            self.obstacles - self.pos,
            axis=1
        )

        if np.min(obstacle_dists) < self.safe_distance:

            collision = True

            collision_penalty = -20.0

        # ==========================================
        # FINAL REWARD
        # ==========================================
        reward = (

            R_progress +
            R_distance +
            R_success +
            collision_penalty

        )

        # ==========================================
        # TERMINATION
        # ==========================================
        terminated = (
            success or collision
        )

        truncated = (
            self.step_count >= self.max_steps
        )

        obs = self._get_obs()

        info = {
            "success": success,
            "collision": collision
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info
        )