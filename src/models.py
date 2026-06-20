"""Neural network modules for value-based RL models."""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """Small multilayer perceptron that estimates Q-values for discrete actions."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = observation_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)
