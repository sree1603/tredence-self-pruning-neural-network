"""
Runs training for three lambda values (low / medium / high), then writes a
summary CSV/markdown table comparing test accuracy vs sparsity level, and
copies the best model's gate-distribution plot to results/best_gate_distribution.png.
"""

import argparse
import json
import os
import shutil

from train import train


def parse_args():
    p = argparse.ArgumentParser(description="Sweep lambda values for the self-pruning network")
    p.add_argument("--lambdas", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threshold", type=float, default=1e-2)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


class Args:
    def __init__(self, lam, epochs, batch_size, lr, threshold, data_dir, out, seed=42):
        self.lam = lam
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.threshold = threshold
        self.data_dir = data_dir
        self.out = out
        self.seed = seed


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    results = []
    for lam in args.lambdas:
        run_out = os.path.join(args.results_dir, f"lam_{lam:g}")
        run_args = Args(lam, args.epochs, args.batch_size, args.lr, args.threshold,
                         args.data_dir, run_out)
        result = train(run_args)
        results.append(result)

    lines = ["| Lambda | Test Accuracy | Sparsity Level (%) |",
             "|---|---|---|"]
    for r in results:
        lines.append(f"| {r['lambda']:g} | {r['final_test_accuracy']:.2f}% | {r['final_sparsity_level_pct']:.2f}% |")
    table_md = "\n".join(lines)

    with open(os.path.join(args.results_dir, "summary_table.md"), "w") as f:
        f.write(table_md + "\n")

    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    baseline_acc = results[0]["final_test_accuracy"]
    candidates = [r for r in results if r["final_test_accuracy"] >= baseline_acc - 5.0]
    best = max(candidates, key=lambda r: r["final_sparsity_level_pct"]) if candidates else results[0]
    best_plot_src = os.path.join(args.results_dir, f"lam_{best['lambda']:g}", "gate_distribution.png")
    best_plot_dst = os.path.join(args.results_dir, "best_gate_distribution.png")
    if os.path.exists(best_plot_src):
        shutil.copy(best_plot_src, best_plot_dst)

    print("\n=== Summary ===")
    print(table_md)
    print(f"\nBest (highest sparsity within 5pt of baseline accuracy): lambda={best['lambda']:g}")
    print(f"Gate distribution plot for best model copied to {best_plot_dst}")


if __name__ == "__main__":
    main()
