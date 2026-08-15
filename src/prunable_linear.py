"""
PrunableLinear: a custom Linear layer that learns to prune its own weights.

Each weight w_ij is paired with a learnable "gate score" g_ij. The gate score
is passed through a sigmoid to produce a gate value in (0, 1), which multiplies
the corresponding weight before the linear transform is applied. Because the
gate is part of the forward computation graph, gradients flow to gate_scores
during backprop exactly like they flow to weight/bias -- no custom autograd
Function is needed, standard autograd handles it.

Driving gate_scores very negative pushes sigmoid(gate_scores) -> 0, which
effectively zeroes out (prunes) that connection's contribution to the output,
without physically removing the parameter (removal/compaction is a separate,
optional step done after training based on the final gate values).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear whose weights can be dynamically
    "gated" (soft-pruned) during training via a learned sigmoid gate per weight.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: if True, adds a learnable bias (bias is NOT gated/pruned --
              only the weight matrix is, matching the spec)
        gate_init: initial value fed into the gate_scores parameter before the
              sigmoid. A positive value (default 3.0 -> sigmoid ~ 0.95) starts
              the network close to "fully open" so it has full capacity at the
              start of training and prunes down from there, rather than
              starting half-pruned (gate_init=0 -> sigmoid=0.5).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 gate_init: float = 3.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), gate_init))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def gates(self) -> torch.Tensor:
        """Return the current gate values (post-sigmoid), shape == weight.shape."""
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of the gate values for this layer. Since gates are always in
        (0, 1) (post-sigmoid), the L1 norm is simply their sum.
        """
        return self.gates().sum()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
