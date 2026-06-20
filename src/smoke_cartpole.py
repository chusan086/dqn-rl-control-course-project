"""Minimal CartPole smoke test for the project environment."""

from __future__ import annotations

import argparse
import random

import gymnasium as gym
import numpy as np
import torch


def run_random_policy(episodes: int, seed: int) -> list[float]:
    """Run a random policy and return episode rewards."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    rewards: list[float] = []

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            done = False
            total_reward = 0.0

            while not done:
                action = env.action_space.sample()
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated

            rewards.append(total_reward)
    finally:
        env.close()

    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short CartPole random-policy smoke test.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rewards = run_random_policy(episodes=args.episodes, seed=args.seed)
    mean_reward = sum(rewards) / len(rewards)

    print(f"CartPole-v1 smoke test passed: episodes={len(rewards)}, mean_reward={mean_reward:.2f}")
    print("episode_rewards=" + ",".join(f"{reward:.1f}" for reward in rewards))


if __name__ == "__main__":
    main()
