"""Shared helpers for experiment scripts."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = False) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dirs() -> None:
    for path in (LOGS_DIR, CHECKPOINTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(float(np.mean(values[start : index + 1])))
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
