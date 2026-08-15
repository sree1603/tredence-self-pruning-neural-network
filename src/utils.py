"""Shared helpers: accuracy evaluation and gate-distribution plotting."""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import SelfPruningNet


@torch.no_grad()
def evaluate_accuracy(model: SelfPruningNet, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def plot_gate_distribution(model: SelfPruningNet, save_path: str, title: str = "Gate value distribution"):
    """
    Histogram of every gate value in the network. A successful run should
    show a large spike near 0 (pruned connections) and a second cluster of
    values away from 0 (kept connections).
    """
    gates = model.all_gate_values().detach().cpu().numpy()

    plt.figure(figsize=(7, 5))
    plt.hist(gates, bins=50, color="#4C72B0", edgecolor="black", alpha=0.85)
    plt.xlabel("Gate value (sigmoid(gate_scores))")
    plt.ylabel("Count")
    plt.title(title)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
