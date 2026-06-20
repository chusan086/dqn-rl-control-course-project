"""Run a random-policy baseline for CartPole-style discrete control tasks."""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np

from utils import LOGS_DIR, ensure_dirs, moving_average, set_seed, write_csv, write_json


def run(args: argparse.Namespace) -> dict[str, object]:
    ensure_dirs()
    set_seed(args.seed)
    env = gym.make(args.env_id)

    tag = f"_{args.tag}" if args.tag else ""
    run_name = f"{args.env_id}{tag}_random_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    csv_path = LOGS_DIR / f"{run_name}.csv"
    summary_path = LOGS_DIR / f"{run_name}_summary.json"

    rewards: list[float] = []
    rows: list[dict[str, object]] = []
    try:
        for episode in range(1, args.episodes + 1):
            _, _ = env.reset(seed=args.seed + episode)
            episode_reward = 0.0
            done = False
            steps = 0

            while not done and steps < args.max_steps:
                action = env.action_space.sample()
                _, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                done = terminated or truncated
                steps += 1

            rewards.append(episode_reward)
            rows.append(
                {
                    "episode": episode,
                    "reward": episode_reward,
                    "moving_avg_reward": moving_average(rewards, args.ma_window)[-1],
                    "epsilon": 1.0,
                    "loss": 0.0,
                    "steps": steps,
                }
            )
    finally:
        env.close()

    write_csv(csv_path, rows, ["episode", "reward", "moving_avg_reward", "epsilon", "loss", "steps"])
    summary = {
        "run_name": run_name,
        "algo": "random",
        "env_id": args.env_id,
        "seed": args.seed,
        "episodes": args.episodes,
        "final_reward": rewards[-1] if rewards else None,
        "mean_reward": float(np.mean(rewards)) if rewards else None,
        "best_moving_avg_reward": max((row["moving_avg_reward"] for row in rows), default=None),
        "csv_path": str(csv_path),
        "checkpoint_path": None,
        "device": "cpu",
    }
    write_json(summary_path, summary)
    print(f"Random baseline saved log: {csv_path}")
    print(f"Random baseline saved summary: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a random-policy baseline.")
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ma-window", type=int, default=20)
    parser.add_argument("--tag", default="", help="Optional label inserted into output file names.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
