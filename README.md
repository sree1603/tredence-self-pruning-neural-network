# Self-Pruning Neural Network

Implementation of a self-pruning feed-forward neural network for CIFAR-10, developed as part of the Tredence AI Engineering Intern case study.

The project explores how a neural network can learn to suppress unnecessary connections during training using **learnable gates** and **L1 sparsity regularization**, instead of relying only on conventional post-training pruning.

## Overview

Neural networks can contain a large number of parameters, which increases memory and computational requirements. Pruning can reduce the number of effective connections by removing parameters that contribute less to the model.

In this project, pruning is incorporated directly into the training process.

Each weight in a custom `PrunableLinear` layer is associated with a learnable `gate_score`. The gate is obtained using a sigmoid transformation:

```text
gate = sigmoid(gate_score)
```

The effective weight used during the forward pass is:

```text
pruned_weight = weight × gate
```

When a gate approaches zero, the corresponding connection becomes effectively inactive.

The training objective is:

```text
Total Loss = Cross Entropy Loss + λ × Sparsity Loss
```

where the sparsity loss is the sum of all sigmoid gate values across the prunable layers.

A larger `λ` places more emphasis on sparsity and therefore creates a stronger pressure to close gates.

---

## Key Features

* Custom `PrunableLinear` layer implemented from scratch
* Learnable `gate_scores` for every weight
* Sigmoid-based differentiable gates
* L1 sparsity regularization
* End-to-end training on CIFAR-10
* Multiple λ experiments
* Test accuracy and sparsity evaluation
* Gate-value distribution visualization
* Unit tests for the pruning mechanism and gradient flow
* Automatic experiment summary and report generation

---

## Model Architecture

The network is a feed-forward MLP operating on flattened CIFAR-10 images.

```text
CIFAR-10 Image
     │
     ▼
Flatten (3 × 32 × 32 = 3072)
     │
     ▼
PrunableLinear (3072 → 1024)
     │
   ReLU
     │
  Dropout
     │
     ▼
PrunableLinear (1024 → 512)
     │
   ReLU
     │
  Dropout
     │
     ▼
PrunableLinear (512 → 256)
     │
   ReLU
     │
  Dropout
     │
     ▼
PrunableLinear (256 → 10)
     │
     ▼
Class logits
```

Every linear layer uses the custom `PrunableLinear` implementation.

---

## PrunableLinear

The custom layer contains:

* Standard weight parameters
* Bias parameters
* A `gate_scores` parameter tensor with the same shape as the weight matrix

During the forward pass:

```python
gates = torch.sigmoid(self.gate_scores)
pruned_weight = self.weight * gates
output = F.linear(x, pruned_weight, self.bias)
```

Because the gates are part of the computation graph, gradients can flow through both the weights and `gate_scores` using standard PyTorch autograd.

The bias is not pruned; only the weight connections are gated.

---

## Sparsity Regularization

The sparsity loss is defined as the L1 norm of the gate values:

```text
Sparsity Loss = Σ |gate|
```

Since sigmoid gates are positive:

```text
Sparsity Loss = Σ gate
```

The complete objective is:

```text
L = L_classification + λ Σ gate
```

The `λ` parameter controls the trade-off between classification performance and sparsity.

* Lower `λ` → weaker pressure toward pruning
* Higher `λ` → stronger pressure toward pruning
* Excessively high `λ` can remove connections that are still useful for classification

---

## Experimental Results

The following experiments were performed using four different λ values.

| Lambda | Test Accuracy | Sparsity Level |
| -----: | ------------: | -------------: |
| 0.0005 |        52.09% |         98.52% |
|  0.001 |        49.54% |         99.64% |
|   0.01 |        37.74% |         99.98% |
|    0.1 |        21.61% |        100.00% |

### Observations

The results show a clear sparsity–accuracy trade-off.

At `λ = 0.0005`, the model achieves the highest test accuracy among the tested configurations while already reaching **98.52% sparsity**.

Increasing λ further increases sparsity:

```text
λ = 0.0005 → 98.52% sparsity
λ = 0.001  → 99.64% sparsity
λ = 0.01   → 99.98% sparsity
λ = 0.1    → 100.00% sparsity
```

However, this additional pruning comes with a substantial decrease in test accuracy.

For the tested configurations, `λ = 0.0005` provides the best accuracy while maintaining very high sparsity. The results therefore demonstrate the expected trade-off between retaining useful connections and aggressively suppressing model parameters.

---

## Gate Distribution

The project generates a histogram of the final gate values.

A successful pruning configuration should show many gate values close to zero, representing effectively pruned connections, together with a remaining group of gates away from zero representing retained connections.

The generated plot is available at:

```text
results/best_gate_distribution.png
```

---

## Project Structure

```text
.
├── README.md
├── REPORT.md
├── train.py
├── run_experiments.py
│
├── src/
│   ├── __init__.py
│   ├── prunable_linear.py
│   ├── model.py
│   ├── data.py
│   └── utils.py
│
├── tests/
│   └── test_prunable_linear.py
│
├── results/
│   ├── summary_table.md
│   ├── summary.json
│   ├── best_gate_distribution.png
│   └── ...
│
└── self_pruning_nn.ipynb
```

---

## Testing

The repository includes lightweight tests covering:

* Output shape of `PrunableLinear`
* Registration of `gate_scores` as a model parameter
* Gradient flow to weights and gate scores
* Gate values remaining within `[0, 1]`
* Zero gates suppressing the corresponding connection
* Correct calculation of the L1 sparsity loss
* Network-level sparsity aggregation
* Reduction of sparsity loss under a strong sparsity penalty

Run the tests with:

```bash
python tests/test_prunable_linear.py
```

---

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a single model

For example:

```bash
python train.py --lam 0.0005 --epochs 30
```

### 3. Run the complete λ sweep

```bash
python run_experiments.py --lambdas 5e-4 1e-3 1e-2 1e-1 --epochs 30
```

The experiment script generates:

* Individual model checkpoints
* Per-run result files
* Gate-distribution plots
* A combined result summary

---

## Dataset

The model is trained and evaluated using the CIFAR-10 dataset provided through `torchvision.datasets`.

The dataset contains 10 image classes and uses 32×32 RGB images.

The training pipeline applies standard CIFAR-10 normalization along with random cropping and horizontal flipping for training augmentation.

---

## Technical Stack

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* CIFAR-10
* PyTorch Autograd
* pytest-compatible unit testing structure

---

## Case Study Objective

This project addresses the core requirements of the Tredence self-pruning neural network case study:

1. Implement a custom prunable linear layer.
2. Associate each weight with a learnable gate.
3. Use sigmoid gates during the forward pass.
4. Add an L1 penalty on gate values.
5. Train on CIFAR-10.
6. Evaluate test accuracy and sparsity.
7. Compare multiple λ values.
8. Analyze the resulting sparsity–accuracy trade-off.
9. Visualize the distribution of final gate values.

---

## Author

**Sri Varshita Sunkara**

This repository contains my implementation and experimental results for the Tredence AI Engineering Intern case study.

