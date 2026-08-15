"""
Train a single SelfPruningNet on CIFAR-10 with a given sparsity weight (lambda).

Total loss = CrossEntropy(logits, y) + lam * total_sparsity_loss
where total_sparsity_loss is the sum of all gate values (post-sigmoid) across
every PrunableLinear layer in the network.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_dataloaders
from src.model import SelfPruningNet
from src.utils import evaluate_accuracy, plot_gate_distribution


def parse_args():
    p = argparse.ArgumentParser(description="Train a self-pruning network on CIFAR-10")
    p.add_argument("--lam", type=float, default=1e-2, help="sparsity loss weight (lambda)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threshold", type=float, default=1e-2, help="gate value below which a weight counts as pruned")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--out", type=str, default="results/run", help="output directory for this run")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train(args):
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)

    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running_cls_loss, running_sparsity_loss, n_batches = 0.0, 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            cls_loss = criterion(logits, y)
            sparsity_loss = model.total_sparsity_loss()
            total_loss = cls_loss + args.lam * sparsity_loss
            total_loss.backward()
            optimizer.step()

            running_cls_loss += cls_loss.item()
            running_sparsity_loss += sparsity_loss.item()
            n_batches += 1

        test_acc = evaluate_accuracy(model, test_loader, device)
        sparsity_level = model.sparsity_level(args.threshold)
        elapsed = time.time() - start

        avg_cls = running_cls_loss / n_batches
        avg_sparsity = running_sparsity_loss / n_batches
        print(f"[lam={args.lam}] epoch {epoch}/{args.epochs} "
              f"cls_loss={avg_cls:.4f} sparsity_term={avg_sparsity:.1f} "
              f"test_acc={test_acc:.2f}% sparsity_level={sparsity_level:.2f}% "
              f"({elapsed:.1f}s)")

        history.append({
            "epoch": epoch,
            "cls_loss": avg_cls,
            "sparsity_term": avg_sparsity,
            "test_acc": test_acc,
            "sparsity_level": sparsity_level,
        })

    final_acc = history[-1]["test_acc"]
    final_sparsity = history[-1]["sparsity_level"]

    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    plot_gate_distribution(model, os.path.join(args.out, "gate_distribution.png"),
                            title=f"Gate distribution (lambda={args.lam})")

    result = {
        "lambda": args.lam,
        "epochs": args.epochs,
        "final_test_accuracy": final_acc,
        "final_sparsity_level_pct": final_sparsity,
        "threshold": args.threshold,
        "history": history,
    }
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nDone. lambda={args.lam} -> test_acc={final_acc:.2f}%, sparsity={final_sparsity:.2f}%")
    print(f"Artifacts saved to {args.out}/")
    return result


if __name__ == "__main__":
    train(parse_args())
