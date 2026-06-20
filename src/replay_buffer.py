"""Experience replay buffer for DQN-style algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch


class ReplayBatch(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


@dataclass
class ReplayBuffer:
    capacity: int
    observation_shape: tuple[int, ...]
    device: torch.device

    def __post_init__(self) -> None:
        self.observations = np.zeros((self.capacity, *self.observation_shape), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, *self.observation_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if self.size < batch_size:
            raise ValueError(f"Cannot sample batch_size={batch_size} from replay size={self.size}")

        indices = np.random.randint(0, self.size, size=batch_size)
        return ReplayBatch(
            observations=torch.as_tensor(self.observations[indices], device=self.device),
            actions=torch.as_tensor(self.actions[indices], device=self.device).unsqueeze(1),
            rewards=torch.as_tensor(self.rewards[indices], device=self.device).unsqueeze(1),
            next_observations=torch.as_tensor(self.next_observations[indices], device=self.device),
            dones=torch.as_tensor(self.dones[indices], device=self.device).unsqueeze(1),
        )

    def __len__(self) -> int:
        return self.size
