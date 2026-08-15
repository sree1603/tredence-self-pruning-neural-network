"""
Lightweight, fast unit tests for PrunableLinear and SelfPruningNet. Use tiny
random tensors (no CIFAR-10 download needed) so they run in seconds.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.prunable_linear import PrunableLinear
from src.model import SelfPruningNet


def test_output_shape():
    layer = PrunableLinear(10, 5)
    x = torch.randn(4, 10)
    out = layer(x)
    assert out.shape == (4, 5), f"expected (4, 5), got {out.shape}"
    print("test_output_shape: OK")


def test_gate_scores_is_parameter():
    layer = PrunableLinear(10, 5)
    param_names = {name for name, _ in layer.named_parameters()}
    assert "gate_scores" in param_names, "gate_scores must be a registered nn.Parameter"
    assert layer.gate_scores.shape == layer.weight.shape
    print("test_gate_scores_is_parameter: OK")


def test_gradients_flow_to_weight_and_gate_scores():
    layer = PrunableLinear(10, 5)
    x = torch.randn(4, 10, requires_grad=False)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert layer.weight.grad is not None, "no gradient reached weight"
    assert not torch.all(layer.weight.grad == 0), "weight gradient is all zero"
    assert layer.gate_scores.grad is not None, "no gradient reached gate_scores"
    assert not torch.all(layer.gate_scores.grad == 0), "gate_scores gradient is all zero"
    print("test_gradients_flow_to_weight_and_gate_scores: OK")


def test_gates_are_bounded_in_unit_interval():
    layer = PrunableLinear(10, 5)
    gates = layer.gates()
    assert torch.all(gates >= 0) and torch.all(gates <= 1), "gates must be in [0, 1]"
    print("test_gates_are_bounded_in_unit_interval: OK")


def test_zero_gate_zeros_that_connection():
    layer = PrunableLinear(3, 1, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.gate_scores.fill_(-50.0)
    x = torch.ones(1, 3)
    out = layer(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-4), \
        f"expected ~0 output when all gates are closed, got {out}"
    print("test_zero_gate_zeros_that_connection: OK")


def test_sparsity_loss_matches_manual_l1():
    layer = PrunableLinear(6, 4)
    manual_l1 = torch.sigmoid(layer.gate_scores).abs().sum()
    assert torch.allclose(layer.sparsity_loss(), manual_l1), "sparsity_loss should equal L1 norm of gates"
    print("test_sparsity_loss_matches_manual_l1: OK")


def test_full_model_forward_and_sparsity_aggregation():
    model = SelfPruningNet(input_dim=3 * 32 * 32, hidden_dims=[64, 32], num_classes=10)
    x = torch.randn(8, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (8, 10)

    total_sparsity = model.total_sparsity_loss()
    manual_total = sum(l.sparsity_loss() for l in model.prunable_layers())
    assert torch.allclose(total_sparsity, manual_total)

    level = model.sparsity_level(threshold=1e-2)
    assert 0.0 <= level <= 100.0
    print(f"test_full_model_forward_and_sparsity_aggregation: OK (initial sparsity_level={level:.2f}%)")


def test_training_step_reduces_sparsity_loss_when_lambda_is_high():
    torch.manual_seed(0)
    model = SelfPruningNet(input_dim=20, hidden_dims=[16], num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    x = torch.randn(16, 20)
    y = torch.randint(0, 2, (16,))
    criterion = torch.nn.CrossEntropyLoss()
    lam = 10.0

    initial_sparsity_loss = model.total_sparsity_loss().item()
    for _ in range(20):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y) + lam * model.total_sparsity_loss()
        loss.backward()
        optimizer.step()
    final_sparsity_loss = model.total_sparsity_loss().item()

    assert final_sparsity_loss < initial_sparsity_loss, \
        f"expected sparsity loss to drop with high lambda, went {initial_sparsity_loss:.2f} -> {final_sparsity_loss:.2f}"
    print(f"test_training_step_reduces_sparsity_loss_when_lambda_is_high: OK "
          f"({initial_sparsity_loss:.2f} -> {final_sparsity_loss:.2f})")


if __name__ == "__main__":
    test_output_shape()
    test_gate_scores_is_parameter()
    test_gradients_flow_to_weight_and_gate_scores()
    test_gates_are_bounded_in_unit_interval()
    test_zero_gate_zeros_that_connection()
    test_sparsity_loss_matches_manual_l1()
    test_full_model_forward_and_sparsity_aggregation()
    test_training_step_reduces_sparsity_loss_when_lambda_is_high()
    print("\nAll tests passed.")
