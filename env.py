from pettingzoo.utils.env import ParallelEnv
import numpy as np
from gymnasium import spaces

class UAVHMARLEnv(ParallelEnv):
    metadata = {"name": "uav_hmarl_v0"}

    def __init__(self, num_uavs=5):
        self.num_uavs = num_uavs
        self.agents = [f"uav_{i}" for i in range(num_uavs)]

        self.r_comm = 6.0
        self.d_safe = 1.0

        self.goal = np.array([10.0, 10.0, 5.0])
        self.tasks = self._generate_tasks()

        # ✅ ADD MAX STEPS
        self.max_steps = 100
        self.step_count = 0

        self.observation_spaces = {
            a: spaces.Box(-20, 20, shape=(12,), dtype=np.float32)
            for a in self.agents
        }

        self.action_spaces = {
            a: spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
            for a in self.agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _generate_tasks(self):
        return np.random.uniform(0, 15, (5, 3))

    def reset(self, seed=None, options=None):
        # UAV states
        self.pos = {a: np.random.uniform(-5, 5, 3) for a in self.agents}
        self.vel = {a: np.zeros(3) for a in self.agents}
        self.energy = {a: 100.0 for a in self.agents}

        # Step counter
        self.step_count = 0

        # Tasks
        self.tasks = self._generate_tasks()
        self.task_assignments = {
            a: i % len(self.tasks) for i, a in enumerate(self.agents)
        }
        return {a: self._obs(a) for a in self.agents}, {}

    
    def _neighbors(self, agent):
        neighbors = []
        for other in self.agents:
            if other == agent:
                continue
            if np.linalg.norm(self.pos[agent] - self.pos[other]) <= self.r_comm:
                neighbors.append(other)
        return neighbors

    def _obs(self, agent):
        neighbors = self._neighbors(agent)

        msg = np.zeros(3)
        if neighbors:
            msg = np.mean([self.pos[n] for n in neighbors], axis=0)

        noise = np.random.normal(0, 0.1, msg.shape)
        task = self.tasks[self.task_assignments[agent]]
        return np.concatenate([
            self.pos[agent],
            self.vel[agent],
            task,
            msg + noise
        ]).astype(np.float32)

    def step(self, actions):
        rewards, terms, truncs, infos = {}, {}, {}, {}

        # ✅ STEP COUNTER
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        for a, act in actions.items():
            act = np.clip(act, -1, 1)
            self.vel[a] = act
            self.pos[a] += act
            self.energy[a] -= np.linalg.norm(act)

        # ---- Reward (normalized & stable) ----

        positions = np.array(list(self.pos.values()))
        dist_goal = np.linalg.norm(positions - self.goal, axis=1)

        # Normalize distance (important)
        R_task = -dist_goal.mean() / 20.0   # scale down

        # Energy penalty (small)
        R_energy = -0.001 * sum(self.energy.values())

        # collision penalty
        R_collision = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
              if np.linalg.norm(positions[i] - positions[j]) < self.d_safe:
                  R_collision -= 1  # reduced penalty

        # coordination reward (scaled)
        R_coord = np.std(positions) / 10.0

        # final reward
        total_reward = (
            3 * R_task +
            1 * R_coord +
            0.2 * R_energy +
            2 * R_collision
        )
        total_reward = total_reward / self.num_uavs

        # ✅ FIX DONE CONDITION
        done = (dist_goal.mean() < 1) or (self.step_count >= self.max_steps)

        for a in self.agents:
            rewards[a] = total_reward
            terms[a] = (dist_goal.mean() < 1) or done
            truncs[a] = False
            infos[a] = {}

        # ✅ CRITICAL FOR RLlib
        terms["__all__"] = done
        truncs["__all__"] = False

        return {a: self._obs(a) for a in self.agents}, rewards, terms, truncs, infos
        
    def render(self):
        import matplotlib.pyplot as plt

        positions = np.array(list(self.pos.values()))

        plt.clf()
        plt.scatter(positions[:, 0], positions[:, 1], label="UAVs")

        for i, task in enumerate(self.tasks):
            plt.scatter(task[0], task[1], marker='x')
            plt.text(task[0], task[1], f"T{i}")

        plt.title(f"Step: {self.step_count}")
        plt.legend()
        plt.pause(0.01)