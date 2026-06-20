"""Run planned control-task experiment suites."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent


def run_command(command: list[str]) -> None:
    print("Running: " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=SRC_DIR.parent)


def parse_seed_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random, DQN, and Double DQN experiments.")
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--tag", default="phase3")
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    python = sys.executable
    seeds = parse_seed_list(args.seeds)

    for seed in seeds:
        run_command(
            [
                python,
                "src/random_baseline.py",
                "--env-id",
                args.env_id,
                "--episodes",
                str(args.episodes),
                "--seed",
                str(seed),
                "--tag",
                args.tag,
            ]
        )

    for algo in ("dqn", "double_dqn"):
        for seed in seeds:
            run_command(
                [
                    python,
                    "src/train_dqn.py",
                    "--env-id",
                    args.env_id,
                    "--episodes",
                    str(args.episodes),
                    "--algo",
                    algo,
                    "--seed",
                    str(seed),
                    "--tag",
                    args.tag,
                    "--log-interval",
                    str(args.log_interval),
                ]
            )


if __name__ == "__main__":
    main()
