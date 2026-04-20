# Tredence_Case_Study_NN_PRUNING
#  Self-Pruning Neural Network (CIFAR-10)

A implementation of a **self-pruning neural network** that learns to remove unnecessary weights during training using learnable gate parameters and sparsity regularization.

---

##  Overview

Traditional pruning removes weights *after* training.  
This project implements a **dynamic self-pruning mechanism** where the network:

- Learns which weights are unimportant
- Removes them during training
- Adapts its architecture automatically

---

##  Key Features

- ✅ Custom `PrunableLinear` layer (no `nn.Linear`)
- ✅ Learnable gate parameters for each weight
- ✅ Hard pruning using Straight-Through Estimator (STE)
- ✅ Sparsity regularization (L1-inspired)
- ✅ Training on CIFAR-10 dataset
- ✅ Visualization of gate distributions
- ✅ Accuracy vs Sparsity analysis

---

##  Architecture

Each weight is associated with a gate:
output = weight × gate × input


- Gates are computed using sigmoid
- Hard thresholding converts them into binary (0 or 1)
- Straight-Through Estimator ensures gradient flow

---

##  Loss Function
Total Loss = Classification Loss + λ × Sparsity Loss


- Classification Loss → CrossEntropy
- Sparsity Loss → Mean of gate values
- λ controls pruning strength

---

## 📊 Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|--------------|--------------|
| 0.001  | 54.40        | 53.71        |
| 0.005  | 54.19        | 53.71        |
| 0.01   | 52.54        | 53.72        |

---

## 📈 Key Observations

- 🔹 The model achieves **~53% sparsity** while maintaining good accuracy
- 🔹 Significant compression with minimal performance loss
- 🔹 Gate values form a **bimodal distribution (0 and 1)**
- 🔹 Hard gating ensures **true pruning**, not soft scaling
- 🔹 Sparsity saturates due to fixed thresholding

---

## 📊 Visualizations

### Gate Distribution
- Clear separation between:
  - ❌ Pruned weights (0)
  - ✅ Important weights (1)

### Sparsity vs Lambda
- Demonstrates saturation effect due to thresholding

---

##  Why L1 Regularization Works

L1 penalty encourages sparsity by pushing gate values toward zero:
Sparsity Loss = Σ |gates|


This leads to:
- Many gates → 0 (pruned)
- Few gates → 1 (important)

---

## 🛠️ Tech Stack

- Python 🐍
- PyTorch 🔥
- NumPy
- Matplotlib

