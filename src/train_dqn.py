"""Train DQN or Double DQN on CartPole-v1."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from models import QNetwork
from replay_buffer import ReplayBuffer
from utils import CHECKPOINTS_DIR, LOGS_DIR, ensure_dirs, get_device, moving_average, set_seed, write_csv, write_json


def epsilon_by_step(step: int, eps_start: float, eps_end: float, eps_decay: float) -> float:
    return eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay)


def select_action(
    policy_net: QNetwork,
    observation: np.ndarray,
    action_dim: int,
    epsilon: float,
    device: torch.device,
) -> int:
    if np.random.random() < epsilon:
        return int(np.random.randint(action_dim))

    with torch.no_grad():
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        return int(policy_net(obs_tensor).argmax(dim=1).item())


def optimize_model(
    algo: str,
    policy_net: QNetwork,
    target_net: QNetwork,
    replay_buffer: ReplayBuffer,
    optimizer: optim.Optimizer,
    batch_size: int,
    gamma: float,
) -> float | None:
    if len(replay_buffer) < batch_size:
        return None

    batch = replay_buffer.sample(batch_size)
    current_q = policy_net(batch.observations).gather(1, batch.actions)

    with torch.no_grad():
        if algo == "double_dqn":
            next_actions = policy_net(batch.next_observations).argmax(dim=1, keepdim=True)
            next_q = target_net(batch.next_observations).gather(1, next_actions)
        else:
            next_q = target_net(batch.next_observations).max(dim=1, keepdim=True).values
        target_q = batch.rewards + gamma * (1.0 - batch.dones) * next_q

    loss = F.smooth_l1_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def train(args: argparse.Namespace) -> dict[str, object]:
    ensure_dirs()
    set_seed(args.seed)
    device = get_device(args.cuda)

    env = gym.make(args.env_id)
    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    policy_net = QNetwork(observation_dim, action_dim, hidden_dims=(args.hidden_dim, args.hidden_dim)).to(device)
    target_net = QNetwork(observation_dim, action_dim, hidden_dims=(args.hidden_dim, args.hidden_dim)).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    replay_buffer = ReplayBuffer(args.buffer_size, env.observation_space.shape, device)
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)

    tag = f"_{args.tag}" if args.tag else ""
    run_name = f"{args.env_id}{tag}_{args.algo}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    csv_path = LOGS_DIR / f"{run_name}.csv"
    summary_path = LOGS_DIR / f"{run_name}_summary.json"
    checkpoint_path = CHECKPOINTS_DIR / f"{run_name}_best.pt"

    episode_rewards: list[float] = []
    rows: list[dict[str, object]] = []
    global_step = 0
    best_ma_reward = -float("inf")

    try:
        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset(seed=args.seed + episode)
            episode_reward = 0.0
            episode_losses: list[float] = []

            for _ in range(args.max_steps):
                epsilon = epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay)
                action = select_action(policy_net, observation, action_dim, epsilon, device)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                replay_buffer.add(observation, action, float(reward), next_observation, done)
                loss = optimize_model(
                    args.algo,
                    policy_net,
                    target_net,
                    replay_buffer,
                    optimizer,
                    args.batch_size,
                    args.gamma,
                )
                if loss is not None:
                    episode_losses.append(loss)

                observation = next_observation
                episode_reward += float(reward)
                global_step += 1

                if global_step % args.target_update_interval == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                if done:
                    break

            episode_rewards.append(episode_reward)
            ma_reward = moving_average(episode_rewards, args.ma_window)[-1]
            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            rows.append(
                {
                    "episode": episode,
                    "reward": episode_reward,
                    "moving_avg_reward": ma_reward,
                    "epsilon": epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay),
                    "loss": avg_loss,
                    "steps": global_step,
                }
            )

            if ma_reward > best_ma_reward and len(replay_buffer) >= args.batch_size:
                best_ma_reward = ma_reward
                torch.save(
                    {
                        "model_state_dict": policy_net.state_dict(),
                        "env_id": args.env_id,
                        "algo": args.algo,
                        "seed": args.seed,
                        "observation_dim": observation_dim,
                        "action_dim": action_dim,
                        "hidden_dim": args.hidden_dim,
                        "episode": episode,
                        "moving_avg_reward": ma_reward,
                    },
                    checkpoint_path,
                )

            if episode % args.log_interval == 0 or episode == args.episodes:
                print(
                    f"episode={episode:04d} reward={episode_reward:.1f} "
                    f"ma{args.ma_window}={ma_reward:.1f} epsilon={rows[-1]['epsilon']:.3f} loss={avg_loss:.4f}"
                )
    finally:
        env.close()

    write_csv(csv_path, rows, ["episode", "reward", "moving_avg_reward", "epsilon", "loss", "steps"])
    summary = {
        "run_name": run_name,
        "algo": args.algo,
        "env_id": args.env_id,
        "seed": args.seed,
        "episodes": args.episodes,
        "final_reward": episode_rewards[-1] if episode_rewards else None,
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else None,
        "best_moving_avg_reward": best_ma_reward if best_ma_reward > -float("inf") else None,
        "csv_path": str(csv_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else None,
        "device": str(device),
    }
    write_json(summary_path, summary)
    print(f"Saved log: {csv_path}")
    print(f"Saved summary: {summary_path}")
    if checkpoint_path.exists():
        print(f"Saved checkpoint: {checkpoint_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN or Double DQN on a Gymnasium control task.")
    parser.add_argument("--algo", choices=["dqn", "double_dqn"], default="dqn")
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=float, default=500.0)
    parser.add_argument("--ma-window", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--tag", default="", help="Optional label inserted into output file names.")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
