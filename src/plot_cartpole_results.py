"""Generate figures and summary tables for control-task experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import EXPERIMENTS_DIR, PROJECT_ROOT


LOGS_DIR = EXPERIMENTS_DIR / "logs"
FIGURES_DIR = EXPERIMENTS_DIR / "figures"
PAPER_FIGURES_DIR = PROJECT_ROOT / "paper_assets" / "figures"
PAPER_TABLES_DIR = PROJECT_ROOT / "paper_assets" / "tables"


def safe_env_name(env_id: str) -> str:
    return env_id.replace("/", "_")


def output_prefix(env_id: str) -> str:
    if env_id == "CartPole-v1":
        return "cartpole"
    return safe_env_name(env_id).lower()


def load_runs(env_id: str, tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    prefix = safe_env_name(env_id)

    for summary_path in sorted(LOGS_DIR.glob(f"{prefix}_{tag}_*_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        csv_path = Path(summary["csv_path"])
        if not csv_path.exists():
            csv_path = LOGS_DIR / f"{summary['run_name']}.csv"
        if not csv_path.exists():
            continue

        frame = pd.read_csv(csv_path)
        frame["algo"] = summary["algo"]
        frame["seed"] = int(summary["seed"])
        frame["run_name"] = summary["run_name"]
        frames.append(frame)
        summaries.append(summary)

    if not frames:
        raise FileNotFoundError(f"No logs found for tag={tag!r} under {LOGS_DIR}")

    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)


def save_reward_curve(data: pd.DataFrame, output_path: Path, env_id: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.figure(figsize=(9, 5.2))
    sns.lineplot(data=data, x="episode", y="moving_avg_reward", hue="algo", errorbar="sd", linewidth=2.2)
    plt.title(f"{env_id} Training Performance")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    if env_id == "CartPole-v1":
        plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_seed_curve(data: pd.DataFrame, output_path: Path, env_id: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    grid = sns.relplot(
        data=data,
        x="episode",
        y="moving_avg_reward",
        hue="seed",
        col="algo",
        kind="line",
        col_wrap=3,
        height=3.2,
        aspect=1.1,
        linewidth=1.6,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels("Episode", "Moving Average Reward")
    grid.set_titles("{col_name}")
    grid.figure.suptitle(f"{env_id} Per-Seed Learning Curves", y=1.03)
    grid.figure.tight_layout()
    grid.figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def summarize(data: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    tail = data.sort_values("episode").groupby(["algo", "seed"]).tail(50)
    tail_mean = tail.groupby(["algo", "seed"], as_index=False)["reward"].mean().rename(columns={"reward": "last50_mean_reward"})

    per_run = summaries[["algo", "seed", "mean_reward", "final_reward", "best_moving_avg_reward"]].merge(
        tail_mean,
        on=["algo", "seed"],
        how="left",
    )

    grouped = (
        per_run.groupby("algo")
        .agg(
            seeds=("seed", "count"),
            mean_reward_mean=("mean_reward", "mean"),
            mean_reward_std=("mean_reward", "std"),
            last50_mean_reward_mean=("last50_mean_reward", "mean"),
            last50_mean_reward_std=("last50_mean_reward", "std"),
            best_moving_avg_reward_mean=("best_moving_avg_reward", "mean"),
            best_moving_avg_reward_std=("best_moving_avg_reward", "std"),
        )
        .reset_index()
    )
    return grouped.round(3)


def save_tables(data: pd.DataFrame, summaries: pd.DataFrame, tag: str) -> None:
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    per_run_path = PAPER_TABLES_DIR / f"{tag}_per_run.csv"
    summary_path = PAPER_TABLES_DIR / f"{tag}_summary.csv"
    markdown_path = PAPER_TABLES_DIR / f"{tag}_summary.md"

    per_run_columns = [
        "run_name",
        "algo",
        "env_id",
        "seed",
        "episodes",
        "final_reward",
        "mean_reward",
        "best_moving_avg_reward",
        "device",
    ]
    summaries.sort_values(["algo", "seed"])[per_run_columns].to_csv(per_run_path, index=False)
    summary = summarize(data, summaries)
    summary.to_csv(summary_path, index=False)
    markdown_path.write_text(to_markdown_table(summary), encoding="utf-8")


def to_markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]) for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot control-task experiment results.")
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--tag", default="phase3")
    args = parser.parse_args()

    data, summaries = load_runs(args.env_id, args.tag)
    prefix = output_prefix(args.env_id)
    curve_path = FIGURES_DIR / f"{prefix}_{args.tag}_moving_average.png"
    seed_curve_path = FIGURES_DIR / f"{prefix}_{args.tag}_per_seed.png"
    save_reward_curve(data, curve_path, args.env_id)
    save_seed_curve(data, seed_curve_path, args.env_id)
    save_tables(data, summaries, f"{prefix}_{args.tag}")

    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(curve_path, PAPER_FIGURES_DIR / curve_path.name)
    shutil.copy2(seed_curve_path, PAPER_FIGURES_DIR / seed_curve_path.name)

    print(f"Saved figure: {curve_path}")
    print(f"Saved figure: {seed_curve_path}")
    print(f"Saved tables under: {PAPER_TABLES_DIR}")


if __name__ == "__main__":
    main()
