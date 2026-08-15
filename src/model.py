"""
SelfPruningNet: a feed-forward classifier for CIFAR-10 built entirely out of
PrunableLinear layers, plus helpers to aggregate sparsity loss / sparsity
level across the whole network.
"""

from typing import List

import torch
import torch.nn as nn

from src.prunable_linear import PrunableLinear


class SelfPruningNet(nn.Module):
    """
    Simple feed-forward (MLP) classifier over flattened CIFAR-10 images
    (3x32x32 = 3072 inputs), using PrunableLinear for every linear layer so
    the whole network can learn to prune itself.
    """

    def __init__(self, input_dim: int = 3 * 32 * 32, hidden_dims: List[int] = (1024, 512, 256),
                 num_classes: int = 10, dropout: float = 0.2):
        super().__init__()

        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(PrunableLinear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        self.hidden = nn.Sequential(*layers)
        self.output = PrunableLinear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        return self.output(x)

    def prunable_layers(self) -> List[PrunableLinear]:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum of the L1 (sum-of-gate-values) sparsity loss across all PrunableLinear layers."""
        return sum(layer.sparsity_loss() for layer in self.prunable_layers())

    def all_gate_values(self) -> torch.Tensor:
        """Flattened tensor of every gate value in the network (post-sigmoid)."""
        return torch.cat([layer.gates().flatten() for layer in self.prunable_layers()])

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        """
        Percentage of weights across all PrunableLinear layers whose gate
        value is below `threshold` (i.e. effectively pruned).
        """
        gates = self.all_gate_values()
        pruned = (gates < threshold).float().sum()
        return (pruned / gates.numel()).item() * 100.0
