"""Configurable PyTorch PINN architecture.

Default: 3 hidden layers x 128 neurons with Tanh, 1-D input (eta),
4-D output [f, theta_f, theta_s, phi]. All topology comes from config.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn


_ACTIVATIONS: Dict[str, nn.Module] = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "swish": nn.SiLU(),
    "elu": nn.ELU(),
    "sigmoid": nn.Sigmoid(),
}


def _get_activation(name: str) -> nn.Module:
    """Return activation module for a config string."""
    key = name.strip().lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation '{name}'. Choose {list(_ACTIVATIONS)}")
    # Return a fresh instance to avoid sharing state.
    cls = type(_ACTIVATIONS[key])
    return cls()


class PINN(nn.Module):
    """Fully-connected PINN mapping eta -> (f, theta_f, theta_s, phi).

    Args:
        input_dim: Input dimension (usually 1 for eta).
        hidden_layers: Neurons per hidden layer, e.g. [128, 128, 128].
        output_dim: Number of outputs (4 for this problem).
        activation: Hidden activation name.
        initialization: Weight init scheme (xavier | kaiming | default).
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_layers: Sequence[int] = (128, 128, 128),
        output_dim: int = 4,
        activation: str = "tanh",
        initialization: str = "xavier",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_layers: List[int] = [int(n) for n in hidden_layers]
        self.output_dim = int(output_dim)
        self.activation_name = activation

        layers: List[nn.Module] = []
        prev = self.input_dim
        act = _get_activation(activation)
        for width in self.hidden_layers:
            layers.append(nn.Linear(prev, width))
            layers.append(type(act)())
            prev = width
        layers.append(nn.Linear(prev, self.output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights(initialization)

    def _init_weights(self, scheme: str) -> None:
        """Initialise linear layers per config scheme."""
        scheme = (scheme or "xavier").lower()
        for module in self.net:
            if isinstance(module, nn.Linear):
                if scheme == "xavier":
                    nn.init.xavier_normal_(module.weight)
                elif scheme == "kaiming":
                    nn.init.kaiming_normal_(
                        module.weight, nonlinearity="tanh"
                    )
                else:
                    nn.init.normal_(module.weight, std=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, eta: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            eta: Tensor of shape (N, 1).

        Returns:
            Tensor of shape (N, 4) with [f, theta_f, theta_s, phi].
        """
        return self.net(eta)

    def predict_components(
        self, eta: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Split raw output into named physical components."""
        out = self.forward(eta)
        return {
            "f": out[:, 0:1],
            "theta_f": out[:, 1:2],
            "theta_s": out[:, 2:3],
            "phi": out[:, 3:4],
        }


def build_pinn(model_cfg: Dict[str, object]) -> PINN:
    """Build a PINN from the ``model`` config block.

    Args:
        model_cfg: Dictionary with hidden_layers, activation, etc.

    Returns:
        Instantiated PINN module.
    """
    return PINN(
        input_dim=int(model_cfg.get("input_dim", 1)),  # type: ignore[arg-type]
        hidden_layers=list(model_cfg.get("hidden_layers", [128, 128, 128])),  # type: ignore[arg-type]
        output_dim=int(model_cfg.get("output_dim", 4)),  # type: ignore[arg-type]
        activation=str(model_cfg.get("activation", "tanh")),
        initialization=str(model_cfg.get("initialization", "xavier")),
    )
