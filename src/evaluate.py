"""Evaluate a saved DQN-family checkpoint."""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import torch

from models import QNetwork
from utils import get_device, set_seed


def evaluate(args: argparse.Namespace) -> list[float]:
    set_seed(args.seed)
    device = get_device(args.cuda)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    env_id = checkpoint.get("env_id", args.env_id)
    hidden_dim = int(checkpoint.get("hidden_dim", args.hidden_dim))
    env = gym.make(env_id)

    observation_dim = int(checkpoint.get("observation_dim", np.prod(env.observation_space.shape)))
    action_dim = int(checkpoint.get("action_dim", env.action_space.n))
    policy_net = QNetwork(observation_dim, action_dim, hidden_dims=(hidden_dim, hidden_dim)).to(device)
    policy_net.load_state_dict(checkpoint["model_state_dict"])
    policy_net.eval()

    rewards: list[float] = []
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            episode_reward = 0.0
            done = False
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(policy_net(obs_tensor).argmax(dim=1).item())
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                done = terminated or truncated
            rewards.append(episode_reward)
    finally:
        env.close()

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    std_reward = float(np.std(rewards)) if rewards else 0.0
    print(
        f"Evaluation: checkpoint={args.checkpoint} episodes={len(rewards)} "
        f"mean_reward={mean_reward:.2f} std_reward={std_reward:.2f}"
    )
    print("episode_rewards=" + ",".join(f"{reward:.1f}" for reward in rewards))
    return rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved DQN checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3026)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
