"""Architecture A evaluation: recurrent MAPPO -> deterministic mid-level -> native PyTorch recurrent SAC.
Evaluation only; does not train or overwrite checkpoints.
This patched version measures persistent all-task completion instead of ending
when the first UAV reaches one assigned task. Since the existing checkpoints
were trained with the earlier early-termination rule, treat these as corrected
evaluation diagnostics, not directly comparable with old evaluation runs.

For a fair later ablation, Architecture B must use the same seeds, checkpoints,
macro interval, environment, success criterion, and MAPPO-action interpretation.
This script preserves the current integration convention: the clipped 3-D MAPPO
output is passed unchanged to planner.macro_to_task_index(). Verify that this
matches the semantics used when the MAPPO checkpoint was trained before using
results as final manuscript evidence.
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
NUM_EPISODES = 30
NUM_UAVS, NUM_TASKS = 7, 5
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
RESULT_DIR = Path(__file__).resolve().parent / "archA_u7_midlevel_evaluation"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def hmarl_env_creator(config):
    return ParallelPettingZooEnv(UAVHMARLEnv(num_uavs=NUM_UAVS, num_tasks=NUM_TASKS))

# Register the same environment name used by the existing MAPPO integration.
register_env("uav_hmarl", hmarl_env_creator)


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


def avoid_completed_task_reassignment(env, proposed_assignments, agents):
    """Prefer uncompleted tasks and maintain unique assignments where possible."""
    completed = set(getattr(env, "completed_tasks", set()))
    available = set(range(len(env.tasks))) - completed
    if not available:
        return dict(proposed_assignments)

    result, used = {}, set()
    # Keep the planner's choices when they are still available and unique.
    for agent in agents:
        idx = int(proposed_assignments[agent])
        if idx in available and idx not in used:
            result[agent] = idx
            used.add(idx)

    # Replace completed/duplicate proposals with the nearest remaining task.
    for agent in agents:
        if agent in result:
            continue
        remaining = available - used
        if not remaining:
            result[agent] = int(proposed_assignments[agent])
            continue
        idx = min(remaining, key=lambda j: float(np.linalg.norm(env.pos[agent] - env.tasks[j])))
        result[agent] = int(idx)
        used.add(idx)
    return result


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


def evaluate_one(ep, seed, mappo, actor, planner):
    env = UAVHMARLEnv(num_uavs=NUM_UAVS, num_tasks=NUM_TASKS)
    obs, _ = env.reset(seed=seed)
    agents = list(env.agents)
    # Keep MAPPO and RSAC recurrent states independent.
    map_states = {}
    rsac_states = {a: actor.initial_state(1, DEVICE) for a in agents}
    prev_actions = {a: np.zeros(3, np.float32) for a in agents}
    prev_rewards = {a: 0.0 for a in agents}
    rewards_hist, collisions_hist, coverage_hist, dist_hist = [], [], [], []
    raw_conflicts, refined_conflicts, raw_unique, refined_unique = [], [], [], []
    raw_dist, refined_dist = [], []
    path = {a: 0.0 for a in agents}
    visited, trajectories = set(), {a: [env.pos[a].copy()] for a in agents}
    success = timeout = False
    steps_done = 0

    for t in range(MAX_STEPS):
        if t % MACRO_INTERVAL == 0:
            macro_actions, raw_idx = {}, {}
            for agent, agent_obs in obs.items():
                pid = "leader_policy" if agent == "uav_0" else "worker_policy"
                policy = mappo.get_policy(pid)
                if policy is None:
                    raise RuntimeError(f"MAPPO checkpoint lacks policy '{pid}'. Check training policy IDs.")
                if agent not in map_states:
                    map_states[agent] = policy.get_initial_state()
                result = mappo.compute_single_action(
                    observation=agent_obs, state=map_states[agent],
                    prev_action=prev_actions[agent], prev_reward=prev_rewards[agent],
                    policy_id=pid, explore=False)
                action, map_states[agent] = result[0], result[1]
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                if action.size < 3:
                    raise ValueError(f"MAPPO action for {agent} has shape {action.shape}; expected 3 values.")
                action = np.clip(action[:3], -1.0, 1.0).astype(np.float32)
                macro_actions[agent] = action
                prev_actions[agent] = action.copy()
                raw_idx[agent] = planner.macro_to_task_index(action, env.tasks)

            raw_ids = list(raw_idx.values())
            raw_conflicts.append(len(raw_ids) - len(set(raw_ids)))
            raw_unique.append(len(set(raw_ids)) / max(1, min(NUM_UAVS, NUM_TASKS)))
            raw_dist.append(float(np.mean([np.linalg.norm(env.pos[a] - env.tasks[raw_idx[a]]) for a in agents])))

            # Mid-level task refinement. The planner returns task coordinates;
            # map them back to task indices before applying them to the environment.
            positions = {a: env.pos[a].copy() for a in agents}
            assigned_positions = planner.assign_tasks(positions, env.tasks.copy(), macro_actions)
            refined_idx = {a: nearest_task_index(assigned_positions[a], env.tasks) for a in agents}
            refined_idx = avoid_completed_task_reassignment(env, refined_idx, agents)
            apply_task_assignments(env, refined_idx)
            refined_ids = list(refined_idx.values())
            refined_conflicts.append(len(refined_ids) - len(set(refined_ids)))
            refined_unique.append(len(set(refined_ids)) / max(1, min(NUM_UAVS, NUM_TASKS)))
            refined_dist.append(float(np.mean([np.linalg.norm(env.pos[a] - env.tasks[refined_idx[a]]) for a in agents])))

        actions, pos_before = {}, {a: env.pos[a].copy() for a in agents}
        for agent in agents:
            # The native RSAC actor expects the RSAC-specific 13-D observation.
            act, rsac_states[agent] = actor.act(env._rsac_obs(agent), state=rsac_states[agent], deterministic=True)
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
        coverage_hist.append(len(visited))
        dist_hist.append(float(np.mean([np.linalg.norm(env.pos[a] - env.tasks[env.task_assignments[a]]) for a in agents])))
        success, timeout = bool(terms.get("__all__", False)), bool(truncs.get("__all__", False))
        if success or timeout:
            break

    reached = sum(int(np.linalg.norm(env.pos[a] - env.tasks[env.task_assignments[a]]) < SUCCESS_RADIUS_M) for a in agents)
    row = {
        "episode": ep, "seed": seed, "steps": steps_done,
        "total_summed_agent_reward": float(np.sum(rewards_hist)),
        "mean_summed_agent_reward_per_step": float(np.mean(rewards_hist)) if rewards_hist else 0.0,
        "mission_success_all_tasks": int(success), "timeout": int(timeout and not success),
        "completed_tasks_count": int(len(getattr(env, "completed_tasks", set()))),
        "task_completion_rate": float(len(getattr(env, "completed_tasks", set())) / max(1, NUM_TASKS)),
        "collision_pair_events": int(np.sum(collisions_hist)),
        "episode_had_collision": int(any(collisions_hist)), "coverage_cells": len(visited),
        "path_length_total_m": float(sum(path.values())),
        "path_length_mean_per_uav_m": float(np.mean(list(path.values()))),
        "task_conflicts_before_refinement_mean": float(np.mean(raw_conflicts)) if raw_conflicts else 0.0,
        "task_conflicts_after_refinement_mean": float(np.mean(refined_conflicts)) if refined_conflicts else 0.0,
        "unique_assignment_rate_before_refinement": float(np.mean(raw_unique)) if raw_unique else 0.0,
        "unique_assignment_rate_after_refinement": float(np.mean(refined_unique)) if refined_unique else 0.0,
        "raw_proposal_mean_task_distance_m": float(np.mean(raw_dist)) if raw_dist else 0.0,
        "refined_assignment_mean_task_distance_m": float(np.mean(refined_dist)) if refined_dist else 0.0,
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
    print("Architecture A: recurrent MAPPO -> deterministic mid-level -> native recurrent SAC")
    print("MAPPO:", mappo_path, "\nRSAC:", rsac_path, "\nDevice:", DEVICE)
    ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
    mappo = None
    try:
        mappo = PPO.from_checkpoint(str(mappo_path))
        actor = load_actor(rsac_path)
        planner = MidLevelPlanner(refine=True, bias_strength=1.0)
        rows, last_traj = [], None
        for ep in range(1, NUM_EPISODES + 1):
            row, last_traj = evaluate_one(ep, BASE_SEED + ep, mappo, actor, planner)
            rows.append(row)
            print(f"Ep {ep:02d}/{NUM_EPISODES} | steps={row['steps']:3d} | reward={row['total_summed_agent_reward']:9.3f} | "
                  f"all_tasks_success={row['mission_success_all_tasks']} | completed={row['completed_tasks_count']}/{NUM_TASKS} | "
                  f"collision events={row['collision_pair_events']:3d} | "
                  f"coverage={row['coverage_cells']:3d} | conflicts raw/refined="
                  f"{row['task_conflicts_before_refinement_mean']:.2f}/{row['task_conflicts_after_refinement_mean']:.2f}")

        with (RESULT_DIR / "archA_episode_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        summary = {"architecture": "A_MAPPO_midlevel_native_RSAC", "episodes": NUM_EPISODES,
                   "mappo_checkpoint": str(mappo_path), "rsac_checkpoint": str(rsac_path)}
        numeric = [k for k, v in rows[0].items() if k not in ("episode", "seed") and isinstance(v, (int, float))]
        for k in numeric:
            vals = np.asarray([r[k] for r in rows], float)
            summary[k + "_mean"] = float(np.mean(vals))
            summary[k + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        with (RESULT_DIR / "archA_summary_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys())); w.writeheader(); w.writerow(summary)

        for field, title, ylabel in [
            ("total_summed_agent_reward", "Architecture A: Team Reward", "Summed agent reward"),
            ("collision_pair_events", "Architecture A: Collision Events", "Colliding UAV-pair events"),
            ("coverage_cells", "Architecture A: Spatial Coverage", "Unique visited grid cells"),
            ("path_length_mean_per_uav_m", "Architecture A: Mean UAV Path Length", "Path length (m)"),
            ("task_conflicts_before_refinement_mean", "Task Conflicts Before Mid-Level", "Duplicate task proposals"),
            ("task_conflicts_after_refinement_mean", "Task Conflicts After Mid-Level", "Duplicate assigned tasks"),
        ]:
            plot_series([r[field] for r in rows], title, ylabel, RESULT_DIR / f"{field}.png")

        # Core mid-level diagnostics: conflicts and assignment distance before/after refinement.
        x = np.arange(1, NUM_EPISODES + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, [r["task_conflicts_before_refinement_mean"] for r in rows], marker="o", label="Before mid-level")
        ax.plot(x, [r["task_conflicts_after_refinement_mean"] for r in rows], marker="s", label="After mid-level")
        ax.set_xlabel("Episode"); ax.set_ylabel("Mean duplicate task assignments")
        ax.set_title("Mid-Level Refinement: Task Conflict Reduction"); ax.grid(True, alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(RESULT_DIR / "midlevel_conflict_reduction.png", dpi=300, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, [r["raw_proposal_mean_task_distance_m"] for r in rows], marker="o", label="Raw MAPPO proposals")
        ax.plot(x, [r["refined_assignment_mean_task_distance_m"] for r in rows], marker="s", label="After mid-level")
        ax.set_xlabel("Episode"); ax.set_ylabel("Mean UAV-to-task distance (m)")
        ax.set_title("Task Distance Before and After Mid-Level Refinement"); ax.grid(True, alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(RESULT_DIR / "midlevel_task_distance_comparison.png", dpi=300, bbox_inches="tight"); plt.close(fig)

        trajectories, tasks = last_traj
        fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
        for agent, points in trajectories.items():
            p = np.asarray(points); ax.plot(p[:, 0], p[:, 1], p[:, 2], label=agent)
        ax.scatter(tasks[:, 0], tasks[:, 1], tasks[:, 2], marker="x", s=80, label="Tasks")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.set_title("Architecture A Trajectories (Final Episode)"); ax.legend(); fig.tight_layout()
        fig.savefig(RESULT_DIR / "archA_trajectories_3d.png", dpi=300, bbox_inches="tight"); plt.close(fig)

        print("\n================ ARCHITECTURE A SUMMARY ================")
        for k in ["mission_success_all_tasks", "completed_tasks_count", "task_completion_rate",
                  "total_summed_agent_reward", "collision_pair_events", "coverage_cells",
                  "task_conflicts_before_refinement_mean", "task_conflicts_after_refinement_mean",
                  "unique_assignment_rate_before_refinement", "unique_assignment_rate_after_refinement",
                  "path_length_mean_per_uav_m"]:
            print(f"{k}: {summary[k + '_mean']:.4f} (SD {summary[k + '_std']:.4f})")
        print("Results saved to:", RESULT_DIR)
    finally:
        if mappo is not None:
            try: mappo.stop()
            except Exception: pass
        ray.shutdown()

if __name__ == "__main__":
    main()
