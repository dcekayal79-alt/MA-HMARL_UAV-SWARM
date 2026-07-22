# uav_hmarl_env.py

from pettingzoo.utils.env import ParallelEnv
import numpy as np
from gymnasium import spaces
from mid import MidLevelPlanner


class UAVHMARLEnv(ParallelEnv):
    metadata = {"name": "uav_hmarl_v2"}

    def __init__(self, num_uavs=5):
        self.num_uavs = num_uavs
        self.agents = [f"uav_{i}" for i in range(num_uavs)]

        # Leader / followers
        self.leader = self.agents[0]
        self.followers = self.agents[1:]

        # Environment params
        self.r_comm = 6.0
        self.d_safe = 1.0
        self.max_steps = 200

        # Tasks
        self.num_tasks = 5
        self.tasks = None
        self.task_assignments = None

        # Mid-level planner
        self.mid_planner = MidLevelPlanner()

        # Observation: pos(3) + vel(3) + task(3) + neighbor msg(3)
        self.observation_spaces = {
            a: spaces.Box(-20, 20, shape=(12,), dtype=np.float32)
            for a in self.agents
        }

        # Actions
        self.action_spaces = {
            self.leader: spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
            **{
                a: spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
                for a in self.followers
            }
        }

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
    def reset(self, seed=None, options=None):
        self.step_count = 0

        self.pos = {a: np.random.uniform(-5, 5, 3) for a in self.agents}
        self.vel = {a: np.zeros(3) for a in self.agents}
        self.energy = {a: 100.0 for a in self.agents}

        self.tasks = np.random.uniform(0, 15, (self.num_tasks, 3))

        self.task_assignments = {
            a: i % self.num_tasks for i, a in enumerate(self.agents)
        }

        return {a: self._obs(a) for a in self.agents}, {}

    # ========================
    # OBSERVATION
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

        noise = np.random.normal(0, 0.02, 3)

        task = self.tasks[self.task_assignments[agent]]

        return np.concatenate([
            self.pos[agent],
            self.vel[agent],
            task,
            msg + noise
        ]).astype(np.float32)

    # ========================
    # STEP
    # ========================
    def step(self, actions):
        self.step_count += 1

        rewards, terms, truncs, infos = {}, {}, {}, {}

        # ========================
        # LEADER CONTROL (TOP LEVEL)
        # ========================
        leader_action = actions[self.leader]

        leader_move = np.clip(leader_action[:3], -1, 1)
        assign_signal = leader_action[3]

        base_task = int(((assign_signal + 1) / 2) * (self.num_tasks - 1))

        for i, a in enumerate(self.agents):
            self.task_assignments[a] = (base_task + i) % self.num_tasks

        # ========================
        # MID-LEVEL PLANNER
        # ========================
        observations = {a: self._obs(a) for a in self.agents}

        assigned_tasks = {
            a: self.tasks[self.task_assignments[a]]
            for a in self.agents
        }

        intents = {
            a: self.mid_planner.compute_intent(observations[a], assigned_tasks[a])
            for a in self.agents
        }

        # ========================
        # APPLY ACTIONS (LOW LEVEL)
        # ========================
        for a, act in actions.items():

            if a == self.leader:
                act = leader_move
            else:
                act = np.clip(act, -1, 1)

                # Blend RL + planner intent
                act = 0.7 * act + 0.3 * intents[a]

            act = 0.5 * act  # stability scaling

            self.vel[a] = act
            self.pos[a] += act
            self.energy[a] -= np.linalg.norm(act)

        # ========================
        # REWARD
        # ========================
        positions = np.array(list(self.pos.values()))

        task_positions = np.array([
            self.tasks[self.task_assignments[a]]
            for a in self.agents
        ])

        dist = np.linalg.norm(positions - task_positions, axis=1)

        R_task = -np.mean(dist) / 20.0
        success_bonus = 2.0 if np.all(dist < 1.0) else 0.0

        energy_used = np.mean([
            np.linalg.norm(self.vel[a]) for a in self.agents
        ])
        R_energy = -0.01 * energy_used

        R_collision = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if np.linalg.norm(positions[i] - positions[j]) < self.d_safe:
                    R_collision -= 1.0

        final_reward = (
            3.0 * R_task +
            2.0 * success_bonus +
            2.0 * R_collision +
            0.05 * R_energy
        )

        final_reward /= self.num_uavs

        # ========================
        # TERMINATION
        # ========================
        success = np.all(dist < 1.0)
        timeout = self.step_count >= self.max_steps

        for a in self.agents:
            rewards[a] = final_reward
            terms[a] = success
            truncs[a] = timeout
            infos[a] = {
                "task": self.task_assignments[a],
                "intent": intents[a]
            }

        terms["__all__"] = success
        truncs["__all__"] = timeout

        return (
            {a: self._obs(a) for a in self.agents},
            rewards,
            terms,
            truncs,
            infos
        )

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

        plt.title(f"Step: {self.step_count}")
        plt.legend()
        plt.pause(0.01)