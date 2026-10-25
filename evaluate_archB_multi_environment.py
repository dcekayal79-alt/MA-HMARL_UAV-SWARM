"""Multi-environment evaluation of Architecture B: recurrent MAPPO -> native PyTorch recurrent SAC, WITHOUT mid-level refinement.
Evaluation only; does not train or overwrite checkpoints.
This evaluation-only script uses the same environment variants, checkpoints, seeds, episode count, and action-to-task decoding convention as the corresponding Architecture A multi-environment script. The deterministic assignment-refinement step is removed: MAPPO's decoded task proposals are passed directly to the environment, and duplicate task assignments are allowed.

The planner helper is used ONLY for the same macro-action-to-task-index decoding as Architecture A; planner.assign_tasks() and completed-task reassignment are not called. This keeps the ablation focused on removal of mid-level refinement.
"""
from pathlib import Path
import csv, random
import numpy as np
import matplotlib.pyplot as plt
import torch
import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPO
from commonhmarlenv_missionfix import UAVHMARLEnv
from rsac_actor import RecurrentSACActor
try:
    from mid_level_task import MidLevelPlanner
except ImportError:
    from mid import MidLevelPlanner

# ------------------------- Settings -------------------------
NUM_EPISODES = 30  # Set to 10 for a quick pilot; use 30 for the main evaluation.
NUM_UAVS, NUM_TASKS = 5, 5  # MAPPO restore-time environment registration only.
UAV_COUNTS = [3, 5, 7]
ENVIRONMENTS = ["baseline", "dense_obstacles", "expanded_area"]
MAX_STEPS, MACRO_INTERVAL = 200, 10
BASE_SEED = 42000
SUCCESS_RADIUS_M = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAPPO_CANDIDATES = [
    Path(r"E:\HMARL\mappo_lstm_final"),
    Path(r"E:\uav_hmarl_project\mappo_lstm_final"),
    Path(r"E:\HMARL_CODE\mappo_lstm_final"),
    Path(r"E:\HMAEL_CODE\mappo_lstm_final"),
]
RSAC_CANDIDATES = [
    Path(r"E:\HMARL\rsac_torch_recurrent_v2\rsac_recurrent_torch_v2_final.pt"),
    Path(r"E:\HMARL_CODE\rsac_torch_recurrent_v2\rsac_recurrent_torch_v2_final.pt"),
    Path(r"E:\HMAEL_CODE\rsac_torch_recurrent_v2\rsac_recurrent_torch_v2_final.pt"),
]
BASE_RESULT_DIR = Path(__file__).resolve().parent / "archB_multi_environment_evaluation"
BASE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR = BASE_RESULT_DIR

def hmarl_env_creator(config):
    return ParallelPettingZooEnv(UAVHMARLEnv(num_uavs=NUM_UAVS, num_tasks=NUM_TASKS))

# Register the same environment name used by the existing MAPPO integration.
register_env("uav_hmarl", hmarl_env_creator)


class ScenarioUAVHMARLEnv(UAVHMARLEnv):
    """Environment variants used to evaluate the proposed mid-level architecture.

    baseline: existing mission-fix environment.
    dense_obstacles: same world and spawn/task ranges, with additional obstacles.
    expanded_area: world_size=30 and UAV/task spawn range expanded from [0,5] to [0,10].
    """
    def __init__(self, num_uavs=5, num_tasks=5, scenario="baseline"):
        super().__init__(num_uavs=num_uavs, num_tasks=num_tasks)
        self.scenario = scenario
        self.spawn_max = 5.0
        if scenario == "dense_obstacles":
            added = np.array([
                [2.0, 3.5, 1.5],
                [3.0, 1.0, 4.0],
                [1.5, 4.5, 3.5],
            ], dtype=np.float32)
            self.obstacles = np.vstack([self.obstacles, added]).astype(np.float32)
            self.num_obstacles = len(self.obstacles)
        elif scenario == "expanded_area":
            self.world_size = 30.0
            self.obstacles = (self.obstacles * 1.5).astype(np.float32)
            self.num_obstacles = len(self.obstacles)
            self.spawn_max = 10.0
        elif scenario != "baseline":
            raise ValueError(f"Unknown scenario: {scenario}")

    def reset(self, *, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        if self.scenario == "expanded_area":
            # Expand both UAV initial positions and task locations, not just world_size.
            self.pos = {
                agent: self.np_random.uniform(0.0, self.spawn_max, size=3).astype(np.float32)
                for agent in self.agents
            }
            self.tasks = self.np_random.uniform(
                0.0, self.spawn_max, size=(self.num_tasks, 3)
            ).astype(np.float32)
            self.task_assignments = {
                agent: i % self.num_tasks for i, agent in enumerate(self.agents)
            }
            self.trajectory_log = {agent: [self.pos[agent].copy()] for agent in self.agents}
            self.prev_dist = None
            observations = {agent: self._obs(agent) for agent in self.agents}
            infos = {agent: {} for agent in self.agents}
        return observations, infos


def first_existing(paths, label):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {label}. Edit candidate paths in this script. Checked:\n" +
                            "\n".join(f"  {p}" for p in paths))


def load_actor(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if ckpt.get("format") != "native_pytorch_recurrent_sac_v2":
        raise ValueError("Expected the native PyTorch RSAC v2 .pt checkpoint, not an RLlib checkpoint.")
    actor = RecurrentSACActor(obs_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"],
                              hidden_size=ckpt["hidden_size"], fc_size=ckpt["fc_size"]).to(DEVICE)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def collision_pairs(env):
    return sum(np.linalg.norm(env.pos[a] - env.pos[b]) < env.safe_distance
               for i, a in enumerate(env.agents) for b in env.agents[i + 1:])


def nearest_task_index(task_position, tasks):
    return int(np.argmin(np.linalg.norm(np.asarray(tasks) - np.asarray(task_position), axis=1)))


def apply_task_assignments(env, assignments):
    """Apply validated task indices; support both updated and older env files.

    Some local copies of commonhmarlenv.py predate set_task_assignments().
    The task_assignments dictionary is initialized by reset() in those versions,
    so update that existing dictionary after validating every index.
    """
    if hasattr(env, "set_task_assignments"):
        env.set_task_assignments(assignments)
        return
    if not hasattr(env, "task_assignments"):
        raise AttributeError(
            "This UAVHMARLEnv has neither set_task_assignments() nor task_assignments. "
            "Please check which commonhmarlenv.py is being imported."
        )
    if not hasattr(env, "tasks"):
        raise RuntimeError("Environment tasks are not initialized; call reset() first.")
    for agent, task_idx in assignments.items():
        if agent not in env.agents:
            raise KeyError(f"Unknown agent in task assignment: {agent}")
        idx = int(task_idx)
        if idx < 0 or idx >= len(env.tasks):
            raise IndexError(f"Task index {idx} is outside [0, {len(env.tasks) - 1}]")
    env.task_assignments.update({agent: int(idx) for agent, idx in assignments.items()})


def plot_series(values, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(values) + 1), values, marker="o", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_one(ep, seed, mappo, actor, decoder, num_uavs, num_tasks, scenario):
    env = ScenarioUAVHMARLEnv(num_uavs=num_uavs, num_tasks=num_tasks, scenario=scenario)
    obs, _ = env.reset(seed=seed)
    agents = list(env.agents)
    map_states = {}
    rsac_states = {a: actor.initial_state(1, DEVICE) for a in agents}
    prev_actions = {a: np.zeros(3, np.float32) for a in agents}
    prev_rewards = {a: 0.0 for a in agents}
    rewards_hist, collisions_hist, dist_hist = [], [], []
    conflict_hist, unique_hist, proposal_dist_hist = [], [], []
    path = {a: 0.0 for a in agents}
    visited, trajectories = set(), {a: [env.pos[a].copy()] for a in agents}
    success = timeout = False
    steps_done = 0

    for t in range(MAX_STEPS):
        if t % MACRO_INTERVAL == 0:
            direct_assignments = {}
            proposal_distances = []
            for agent, agent_obs in obs.items():
                policy_id = "leader_policy" if agent == "uav_0" else "worker_policy"
                policy = mappo.get_policy(policy_id)
                if policy is None:
                    raise RuntimeError(f"MAPPO checkpoint lacks policy '{policy_id}'. Check policy IDs.")
                if agent not in map_states:
                    map_states[agent] = policy.get_initial_state()
                result = mappo.compute_single_action(
                    observation=agent_obs, state=map_states[agent],
                    prev_action=prev_actions[agent], prev_reward=prev_rewards[agent],
                    policy_id=policy_id, explore=False)
                action, map_states[agent] = result[0], result[1]
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                if action.size < 3:
                    raise ValueError(f"MAPPO action for {agent} has shape {action.shape}; expected 3 values.")
                action = np.clip(action[:3], -1.0, 1.0).astype(np.float32)
                prev_actions[agent] = action.copy()

                # Match Architecture A's proposal decoding, but do not refine assignments.
                idx = int(decoder.macro_to_task_index(action, env.tasks))
                direct_assignments[agent] = idx
                proposal_distances.append(float(np.linalg.norm(env.pos[agent] - env.tasks[idx])))

            # Ablation: apply raw decoded proposals directly; duplicate assignments are allowed.
            apply_task_assignments(env, direct_assignments)
            ids = list(direct_assignments.values())
            mean_conflicts = float(len(ids) - len(set(ids)))
            mean_unique = float(len(set(ids)) / max(1, min(num_uavs, num_tasks)))
            conflict_hist.append(mean_conflicts)
            unique_hist.append(mean_unique)
            proposal_dist_hist.append(float(np.mean(proposal_distances)) if proposal_distances else 0.0)

        actions, pos_before = {}, {a: env.pos[a].copy() for a in agents}
        for agent in agents:
            act, rsac_states[agent] = actor.act(
                env._rsac_obs(agent), state=rsac_states[agent], deterministic=True)
            actions[agent] = np.clip(np.asarray(act, np.float32).reshape(3), -1.0, 1.0)

        obs, rewards, terms, truncs, _ = env.step(actions)
        steps_done = t + 1
        for a in agents:
            prev_rewards[a] = float(rewards.get(a, 0.0))
            path[a] += float(np.linalg.norm(env.pos[a] - pos_before[a]))
            trajectories[a].append(env.pos[a].copy())
            visited.add(tuple(np.floor(env.pos[a]).astype(int)))
        rewards_hist.append(float(sum(rewards.values())))
        collisions_hist.append(int(collision_pairs(env)))
        dist_hist.append(float(np.mean([
            np.linalg.norm(env.pos[a] - env.tasks[env.task_assignments[a]]) for a in agents
        ])))
        success = bool(terms.get("__all__", False))
        timeout = bool(truncs.get("__all__", False))
        if success or timeout:
            break

    reached = sum(int(np.linalg.norm(env.pos[a] - env.tasks[env.task_assignments[a]]) < SUCCESS_RADIUS_M)
                  for a in agents)
    completed = len(getattr(env, "completed_tasks", set()))
    mean_conflicts = float(np.mean(conflict_hist)) if conflict_hist else 0.0
    mean_unique = float(np.mean(unique_hist)) if unique_hist else 0.0
    mean_proposal_distance = float(np.mean(proposal_dist_hist)) if proposal_dist_hist else 0.0
    row = {
        "episode": ep, "seed": seed, "steps": steps_done,
        "total_summed_agent_reward": float(np.sum(rewards_hist)),
        "mean_summed_agent_reward_per_step": float(np.mean(rewards_hist)) if rewards_hist else 0.0,
        "mission_success_all_tasks": int(success), "timeout": int(timeout and not success),
        "completed_tasks_count": int(completed),
        "task_completion_rate": float(completed / max(1, num_tasks)),
        "collision_pair_events": int(np.sum(collisions_hist)),
        "episode_had_collision": int(any(collisions_hist)),
        "coverage_cells": len(visited),
        "path_length_total_m": float(sum(path.values())),
        "path_length_mean_per_uav_m": float(np.mean(list(path.values()))),
        # No refinement in Architecture B: these shared fields duplicate the direct proposal metrics.
        "task_conflicts_before_refinement_mean": mean_conflicts,
        "task_conflicts_after_refinement_mean": mean_conflicts,
        "unique_assignment_rate_before_refinement": mean_unique,
        "unique_assignment_rate_after_refinement": mean_unique,
        "raw_proposal_mean_task_distance_m": mean_proposal_distance,
        "refined_assignment_mean_task_distance_m": mean_proposal_distance,
        "mean_distance_to_current_assigned_task_m": float(np.mean(dist_hist)) if dist_hist else 0.0,
        "assigned_tasks_reached_at_episode_end": int(reached),
    }
    traj = (trajectories, env.tasks.copy())
    env.close()
    return row, traj


def main():
    random.seed(BASE_SEED); np.random.seed(BASE_SEED); torch.manual_seed(BASE_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(BASE_SEED)
    mappo_path = first_existing(MAPPO_CANDIDATES, "MAPPO checkpoint")
    rsac_path = first_existing(RSAC_CANDIDATES, "native PyTorch RSAC checkpoint")
    print("Architecture B multi-environment evaluation: MAPPO -> recurrent SAC WITHOUT mid-level refinement")
    print("MAPPO:", mappo_path, "\nRSAC:", rsac_path, "\nDevice:", DEVICE)
    print(f"UAV counts={UAV_COUNTS}; environments={ENVIRONMENTS}; episodes/config={NUM_EPISODES}; tasks=5")
    print("Scenario definitions: baseline; dense_obstacles (+3 obstacles); expanded_area (world=30, spawn/task range=[0,10]).")
    print("Direct assignments use the same macro-action decoder as Architecture A; duplicates are intentionally allowed.")
    ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
    mappo = None
    combined_summaries = []
    try:
        mappo = PPO.from_checkpoint(str(mappo_path))
        actor = load_actor(rsac_path)
        # Used only to decode MAPPO macro-actions into task indices; assign_tasks() is never called.
        decoder = MidLevelPlanner(refine=True, bias_strength=1.0)
        for scenario in ENVIRONMENTS:
            for num_uavs in UAV_COUNTS:
                num_tasks = 5
                result_dir = BASE_RESULT_DIR / f"{scenario}_u{num_uavs}"
                result_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n===== Scenario={scenario} | UAVs={num_uavs} | Tasks={num_tasks} | Episodes={NUM_EPISODES} =====")
                rows, last_traj = [], None
                for ep in range(1, NUM_EPISODES + 1):
                    row, last_traj = evaluate_one(ep, BASE_SEED + ep, mappo, actor, decoder,
                                                  num_uavs, num_tasks, scenario)
                    row["scenario"] = scenario
                    row["num_uavs"] = num_uavs
                    row["num_tasks"] = num_tasks
                    rows.append(row)
                    print(f"Ep {ep:02d}/{NUM_EPISODES} | steps={row['steps']:3d} | reward={row['total_summed_agent_reward']:10.3f} | "
                          f"completed={row['completed_tasks_count']}/{num_tasks} | collisions={row['collision_pair_events']:3d} | "
                          f"coverage={row['coverage_cells']:3d} | direct conflicts={row['task_conflicts_before_refinement_mean']:.2f}")

                with (result_dir / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
                summary = {
                    "architecture": "B_MAPPO_direct_RSAC_without_midlevel",
                    "scenario": scenario, "num_uavs": num_uavs, "num_tasks": num_tasks,
                    "episodes": NUM_EPISODES, "mappo_checkpoint": str(mappo_path),
                    "rsac_checkpoint": str(rsac_path),
                }
                numeric = [k for k, v in rows[0].items()
                           if k not in ("episode", "seed", "scenario", "num_uavs", "num_tasks") and isinstance(v, (int, float))]
                for k in numeric:
                    vals = np.asarray([r[k] for r in rows], float)
                    summary[k + "_mean"] = float(np.mean(vals))
                    summary[k + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                with (result_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(summary.keys())); w.writeheader(); w.writerow(summary)
                combined_summaries.append(summary)

                for field, title, ylabel in [
                    ("total_summed_agent_reward", "Team Reward", "Summed agent reward"),
                    ("collision_pair_events", "Collision Events", "Colliding UAV-pair events"),
                    ("coverage_cells", "Spatial Coverage", "Unique visited grid cells"),
                    ("path_length_mean_per_uav_m", "Mean UAV Path Length", "Path length (m)"),
                    ("task_conflicts_before_refinement_mean", "Direct Assignment Conflicts", "Duplicate task proposals"),
                    ("task_completion_rate", "Task Completion Rate", "Completed tasks / total tasks"),
                ]:
                    plot_series([r[field] for r in rows], f"Architecture B: {scenario}, {num_uavs} UAVs — {title}",
                                ylabel, result_dir / f"{field}.png")

                trajectories, tasks = last_traj
                fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
                for agent, points in trajectories.items():
                    p = np.asarray(points); ax.plot(p[:, 0], p[:, 1], p[:, 2], label=agent)
                ax.scatter(tasks[:, 0], tasks[:, 1], tasks[:, 2], marker="x", s=80, label="Tasks")
                ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
                ax.set_title(f"Architecture B: {scenario}, {num_uavs} UAVs (final episode)")
                ax.legend(); fig.tight_layout()
                fig.savefig(result_dir / "trajectories_3d.png", dpi=300, bbox_inches="tight"); plt.close(fig)
                print("Scenario results saved to:", result_dir)

        combined_path = BASE_RESULT_DIR / "combined_environment_summary.csv"
        if combined_summaries:
            with combined_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(combined_summaries[0].keys()))
                w.writeheader(); w.writerows(combined_summaries)
        print("\n================ ALL SCENARIOS COMPLETE ================")
        print("Combined summary:", combined_path)
        print("All outputs:", BASE_RESULT_DIR)
        print("Reminder: report actual measured values only; Architecture B does not apply mid-level refinement.")
    finally:
        if mappo is not None:
            try: mappo.stop()
            except Exception: pass
        ray.shutdown()

if __name__ == "__main__":
    main()
