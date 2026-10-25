"""Architecture B ablation: recurrent MAPPO -> native PyTorch recurrent SAC.
WITHOUT the deterministic mid-level refinement module.

Uses the current mission-fix environment and current native PyTorch RSAC v2
checkpoint. Direct MAPPO proposals are mapped to nearest task indices exactly
as in the current Architecture A evaluator; duplicate assignments are allowed.
Only the mid-level refinement/conflict-resolution step is removed.
"""
from pathlib import Path
import csv, random, inspect
import numpy as np
import matplotlib.pyplot as plt
import torch
import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPO
from commonhmarlenv_missionfix import UAVHMARLEnv
from rsac_actor import RecurrentSACActor

# Keep these identical to the current Architecture A evaluator.
NUM_EPISODES = 100
NUM_UAVS, NUM_TASKS = 5, 5
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
RESULT_DIR = Path(__file__).resolve().parent / "archB_without_midlevel_evaluation"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def env_creator(config):
    return ParallelPettingZooEnv(UAVHMARLEnv(num_uavs=NUM_UAVS, num_tasks=NUM_TASKS))

register_env("uav_hmarl", env_creator)


def first_existing(paths, label):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {label}; edit candidates:\n" + "\n".join(map(str, paths)))


def load_actor(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if ckpt.get("format") != "native_pytorch_recurrent_sac_v2":
        raise ValueError("Expected native PyTorch recurrent SAC v2 checkpoint, not an RLlib SAC checkpoint.")
    actor = RecurrentSACActor(
        obs_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"],
        hidden_size=ckpt["hidden_size"], fc_size=ckpt["fc_size"]
    ).to(DEVICE)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def nearest_task_index(action, tasks):
    """Use the same task-proposal interpretation as current Architecture A."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    tasks = np.asarray(tasks, dtype=np.float32)
    if action.size < 3 or tasks.ndim != 2 or tasks.shape[1] != 3 or not len(tasks):
        raise ValueError(f"Invalid action/tasks shapes: {action.shape}, {tasks.shape}")
    return int(np.argmin(np.linalg.norm(tasks - action[:3], axis=1)))


def apply_direct_assignments(env, assignments):
    """Apply raw MAPPO task proposals. Deliberately do not enforce uniqueness."""
    if hasattr(env, "set_task_assignments"):
        env.set_task_assignments(assignments)
        return
    if not hasattr(env, "task_assignments"):
        raise AttributeError("Environment has no task_assignments dictionary; check imported environment.")
    for agent, idx in assignments.items():
        if agent not in env.agents or not 0 <= int(idx) < len(env.tasks):
            raise ValueError(f"Invalid assignment: {agent} -> {idx}")
    env.task_assignments.update({a: int(i) for a, i in assignments.items()})


def collision_pairs(env):
    return sum(np.linalg.norm(env.pos[a] - env.pos[b]) < env.safe_distance
               for i, a in enumerate(env.agents) for b in env.agents[i + 1:])


def plot_series(values, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(values) + 1), values, marker="o", linewidth=1.2)
    ax.set_xlabel("Episode"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig)


def evaluate_one(episode, seed, mappo, actor):
    env = UAVHMARLEnv(num_uavs=NUM_UAVS, num_tasks=NUM_TASKS)
    obs, _ = env.reset(seed=seed)
    agents = list(env.agents)
    map_states = {}
    rsac_states = {a: actor.initial_state(1, DEVICE) for a in agents}
    prev_actions = {a: np.zeros(3, np.float32) for a in agents}
    prev_rewards = {a: 0.0 for a in agents}
    reward_hist, collision_hist, coverage_hist, distance_hist = [], [], [], []
    conflict_hist, unique_hist, proposal_dist_hist = [], [], []
    path = {a: 0.0 for a in agents}
    visited, trajectories = set(), {a: [env.pos[a].copy()] for a in agents}
    success = timeout = False
    steps_done = 0

    for t in range(MAX_STEPS):
        # MAPPO proposes tasks every macro interval; no MidLevelPlanner is called.
        if t % MACRO_INTERVAL == 0:
            direct_assignments, proposal_distances = {}, []
            for agent, agent_obs in obs.items():
                policy_id = "leader_policy" if agent == "uav_0" else "worker_policy"
                policy = mappo.get_policy(policy_id)
                if policy is None:
                    raise RuntimeError(f"MAPPO checkpoint lacks policy '{policy_id}'.")
                if agent not in map_states:
                    map_states[agent] = policy.get_initial_state()
                result = mappo.compute_single_action(
                    observation=agent_obs, state=map_states[agent],
                    prev_action=prev_actions[agent], prev_reward=prev_rewards[agent],
                    policy_id=policy_id, explore=False
                )
                action, map_states[agent] = result[0], result[1]
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                if action.size < 3:
                    raise ValueError(f"MAPPO action for {agent} has shape {action.shape}; expected 3 values.")
                action = np.clip(action[:3], -1.0, 1.0).astype(np.float32)
                prev_actions[agent] = action.copy()
                idx = nearest_task_index(action, env.tasks)
                direct_assignments[agent] = idx
                proposal_distances.append(float(np.linalg.norm(env.pos[agent] - env.tasks[idx])))

            # Crucial ablation difference: raw assignments go directly to env.
            # Duplicate task assignments are intentionally retained.
            apply_direct_assignments(env, direct_assignments)
            ids = list(direct_assignments.values())
            conflict_hist.append(len(ids) - len(set(ids)))
            unique_hist.append(len(set(ids)) / max(1, min(NUM_UAVS, NUM_TASKS)))
            proposal_dist_hist.append(float(np.mean(proposal_distances)))

        actions, pos_before = {}, {a: env.pos[a].copy() for a in agents}
        for agent in agents:
            act, rsac_states[agent] = actor.act(
                env._rsac_obs(agent), state=rsac_states[agent], deterministic=True
            )
            actions[agent] = np.clip(np.asarray(act, np.float32).reshape(3), -1.0, 1.0)

        obs, rewards, terms, truncs, _ = env.step(actions)
        steps_done = t + 1
        for a in agents:
            prev_rewards[a] = float(rewards.get(a, 0.0))
            path[a] += float(np.linalg.norm(env.pos[a] - pos_before[a]))
            trajectories[a].append(env.pos[a].copy())
            visited.add(tuple(np.floor(env.pos[a]).astype(int)))
        reward_hist.append(float(sum(rewards.values())))
        collision_hist.append(int(collision_pairs(env)))
        coverage_hist.append(len(visited))
        distance_hist.append(float(np.mean([
            np.linalg.norm(env.pos[a] - env.tasks[env.task_assignments[a]]) for a in agents
        ])))
        success, timeout = bool(terms.get("__all__", False)), bool(truncs.get("__all__", False))
        if success or timeout:
            break

    reached = sum(int(np.linalg.norm(env.pos[a] - env.tasks[env.task_assignments[a]]) < SUCCESS_RADIUS_M)
                  for a in agents)
    completed = len(getattr(env, "completed_tasks", set()))
    # Shared columns retained to make comparison with Architecture A straightforward.
    # In this baseline, "before" and "after" are identical because no refinement occurs.
    mean_conflicts = float(np.mean(conflict_hist)) if conflict_hist else 0.0
    mean_unique = float(np.mean(unique_hist)) if unique_hist else 0.0
    row = {
        "episode": episode, "seed": seed, "steps": steps_done,
        "total_summed_agent_reward": float(np.sum(reward_hist)),
        "mean_summed_agent_reward_per_step": float(np.mean(reward_hist)) if reward_hist else 0.0,
        "mission_success_all_tasks": int(success), "timeout": int(timeout and not success),
        "completed_tasks_count": int(completed), "task_completion_rate": float(completed / max(1, NUM_TASKS)),
        "collision_pair_events": int(np.sum(collision_hist)), "episode_had_collision": int(any(collision_hist)),
        "coverage_cells": len(visited), "path_length_total_m": float(sum(path.values())),
        "path_length_mean_per_uav_m": float(np.mean(list(path.values()))),
        "task_conflicts_before_refinement_mean": mean_conflicts,
        "task_conflicts_after_refinement_mean": mean_conflicts,
        "unique_assignment_rate_before_refinement": mean_unique,
        "unique_assignment_rate_after_refinement": mean_unique,
        "raw_proposal_mean_task_distance_m": float(np.mean(proposal_dist_hist)) if proposal_dist_hist else 0.0,
        "refined_assignment_mean_task_distance_m": float(np.mean(proposal_dist_hist)) if proposal_dist_hist else 0.0,
        "mean_distance_to_current_assigned_task_m": float(np.mean(distance_hist)) if distance_hist else 0.0,
        "assigned_tasks_reached_at_episode_end": int(reached),
    }
    trajectory_data = (trajectories, env.tasks.copy())
    env.close()
    return row, trajectory_data


def main():
    random.seed(BASE_SEED); np.random.seed(BASE_SEED); torch.manual_seed(BASE_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(BASE_SEED)
    mappo_path = first_existing(MAPPO_CANDIDATES, "MAPPO checkpoint")
    rsac_path = first_existing(RSAC_CANDIDATES, "native PyTorch RSAC checkpoint")
    print("=" * 68)
    print("ARCHITECTURE B: MAPPO -> RSAC (WITHOUT MID-LEVEL)")
    print("MAPPO:", mappo_path, "\nRSAC:", rsac_path)
    print("Environment module:", inspect.getfile(UAVHMARLEnv))
    print("RSAC actor module:", inspect.getfile(RecurrentSACActor))
    print(f"Episodes={NUM_EPISODES}, UAVs={NUM_UAVS}, tasks={NUM_TASKS}, max_steps={MAX_STEPS}, macro_interval={MACRO_INTERVAL}")
    print("Device:", DEVICE)
    print("=" * 68)

    ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
    mappo = None
    try:
        mappo = PPO.from_checkpoint(str(mappo_path))
        actor = load_actor(rsac_path)
        rows, last_trajectory = [], None
        for ep in range(1, NUM_EPISODES + 1):
            row, last_trajectory = evaluate_one(ep, BASE_SEED + ep, mappo, actor)
            rows.append(row)
            print(f"Ep {ep:03d}/{NUM_EPISODES} | steps={row['steps']:3d} | reward={row['total_summed_agent_reward']:9.3f} | "
                  f"success={row['mission_success_all_tasks']} | completed={row['completed_tasks_count']}/{NUM_TASKS} | "
                  f"collisions={row['collision_pair_events']:3d} | coverage={row['coverage_cells']:3d} | "
                  f"direct conflicts={row['task_conflicts_before_refinement_mean']:.2f}")

        episode_csv = RESULT_DIR / "archB_episode_metrics.csv"
        with episode_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        summary = {
            "architecture": "B_MAPPO_direct_native_RSAC_without_midlevel", "episodes": NUM_EPISODES,
            "mappo_checkpoint": str(mappo_path), "rsac_checkpoint": str(rsac_path),
            "environment_module": inspect.getfile(UAVHMARLEnv),
        }
        numeric = [k for k, v in rows[0].items() if k not in ("episode", "seed") and isinstance(v, (int, float))]
        for key in numeric:
            vals = np.asarray([r[key] for r in rows], float)
            summary[key + "_mean"] = float(np.mean(vals))
            summary[key + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary_csv = RESULT_DIR / "archB_summary_metrics.csv"
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys())); w.writeheader(); w.writerow(summary)

        for field, title, ylabel in [
            ("total_summed_agent_reward", "Architecture B: Team Reward", "Summed agent reward"),
            ("collision_pair_events", "Architecture B: Collision Events", "Colliding UAV-pair events"),
            ("coverage_cells", "Architecture B: Spatial Coverage", "Unique visited grid cells"),
            ("path_length_mean_per_uav_m", "Architecture B: Mean UAV Path Length", "Path length (m)"),
            ("task_conflicts_before_refinement_mean", "Architecture B: Direct Assignment Conflicts", "Duplicate task proposals"),
            ("task_completion_rate", "Architecture B: Task Completion Rate", "Completed tasks / total tasks"),
        ]:
            plot_series([r[field] for r in rows], title, ylabel, RESULT_DIR / f"{field}.png")

        trajectories, tasks = last_trajectory
        fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
        for agent, points in trajectories.items():
            p = np.asarray(points); ax.plot(p[:, 0], p[:, 1], p[:, 2], label=agent)
        ax.scatter(tasks[:, 0], tasks[:, 1], tasks[:, 2], marker="x", s=80, label="Tasks")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.set_title("Architecture B Trajectories (Final Episode)"); ax.legend(); fig.tight_layout()
        fig.savefig(RESULT_DIR / "archB_trajectories_3d.png", dpi=300, bbox_inches="tight"); plt.close(fig)

        print("\n================ ARCHITECTURE B SUMMARY ================")
        for key in ["mission_success_all_tasks", "completed_tasks_count", "task_completion_rate",
                    "total_summed_agent_reward", "collision_pair_events", "coverage_cells",
                    "task_conflicts_before_refinement_mean", "unique_assignment_rate_before_refinement",
                    "path_length_mean_per_uav_m"]:
            print(f"{key}: {summary[key + '_mean']:.4f} (SD {summary[key + '_std']:.4f})")
        print("Episode metrics:", episode_csv); print("Summary metrics:", summary_csv)
        print("Plots and trajectory saved in:", RESULT_DIR)
    finally:
        if mappo is not None:
            try: mappo.stop()
            except Exception: pass
        ray.shutdown()


if __name__ == "__main__":
    main()
