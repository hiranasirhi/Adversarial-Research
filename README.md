# Adversarial-Research
# LARAR: Layer-wise Adversarial Robustness using Adaptive Regularization

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: UNSW-NB15](https://img.shields.io/badge/Dataset-UNSW--NB15-orange.svg)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

> A layer-wise adversarial training framework that improves robustness of deep learning-based Network Intrusion Detection Systems (NIDS) using **Layer Vulnerability Score (LVS)**, **adaptive regularization**, and **auxiliary classifiers**.

## Overview

LARAR is an adversarial defense framework for deep learning-based Network Intrusion Detection Systems (NIDS). Unlike existing methods that treat neural networks as black boxes and apply uniform regularization across all layers, LARAR introduces **layer-wise vulnerability analysis** and **adaptive regularization** to intelligently allocate defensive resources where they matter most.

The key idea is simple: not all layers of a neural network are equally vulnerable to adversarial attacks. LARAR quantifies this vulnerability at each layer using the **Layer Vulnerability Score (LVS)** metric and uses learnable per-layer weights to strengthen defenses at the most susceptible layers.

---

## Key Contributions

- **Layer Vulnerability Score (LVS):** A new interpretable metric that quantifies how much each hidden layer's activations are perturbed by adversarial inputs, normalized by clean activations.
- **Adaptive Layer-wise Regularization:** Learnable per-layer weights that automatically focus regularization on the most vulnerable layers during training.
- **Multi-level Supervision via Auxiliary Classifiers:** Additional classifiers attached to intermediate layers that provide early adversarial detection capability before reaching the final output layer.
- **Principled Detection Thresholds:** Statistically grounded per-layer thresholds derived from clean validation data for real-time adversarial input detection.

---

## Architecture

```
Input (d=35 features)
        │
   ┌────▼────┐
   │  FC1    │  128 units, ReLU, BatchNorm
   │  + Hook │◄── LVS Monitor + Auxiliary Classifier (g1)
   └────┬────┘
        │
   ┌────▼────┐
   │  FC2    │  64 units, ReLU, BatchNorm
   │  + Hook │◄── LVS Monitor + Auxiliary Classifier (g2)
   └────┬────┘
        │
   ┌────▼────┐
   │ Output  │  1 unit, Sigmoid (binary classification)
   └─────────┘
```


## Requirements

```
Python >= 3.8
PyTorch >= 2.0
NumPy
Pandas
sklearn (LabelEncoder, StandardScaler)
torch
seaborn
scikit-learn
matplotlib
pyarrow
```

## Dataset

This project uses the **UNSW-NB15** dataset, a comprehensive network intrusion detection benchmark.

- **Download:** [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **Total samples:** 82,332
- **Train / Test split:** 70% / 30% (stratified)
- **Features used:** 35 (after preprocessing)
- **Classes:** Normal traffic (0) vs. Attack traffic (1)

---

## Usage

### Training LARAR

```bash
python train.py --epochs 20 --batch_size 64 --lr 0.001 --epsilon 0.3 --pgd_steps 10
```

### Evaluating against adversarial attacks

```bash
python evaluate.py --model_path checkpoints/larar_best.pt --attack fgsm --epsilon 0.3
python evaluate.py --model_path checkpoints/larar_best.pt --attack pgd --epsilon 0.3
python evaluate.py --model_path checkpoints/larar_best.pt --attack transfer
```

### Running the full pipeline

```bash
python hybrid.py
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Hidden Layer 1 | 128 units |
| Hidden Layer 2 | 64 units |
| Activation | ReLU |
| Normalization | Batch Normalization |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Epochs | 20 |
| PGD Iterations | 10 |
| PGD Step Size (α) | 0.01 |
| ε (perturbation budget) | 0.3 (curriculum: 0.0 → 0.3) |
| λ_aux | 0.2 |
| λ_GA | 1.0 |
| λ_FS | 0.5 |
| β (LVS scaling) | 0.3 |
| Detection threshold k | 2.5 |
| Safety margin λ | 1.2 |

---

## Layer Vulnerability Analysis

The Layer Vulnerability Score (LVS) is a per-layer interpretability metric introduced in LARAR that quantifies how much each hidden layer's internal representations are disrupted by adversarial perturbations. It is computed as the batch-averaged relative difference between clean and adversarial activations at each layer, normalized to be scale-invariant across layers of different dimensionalities. A high LVS at a given layer indicates that the layer significantly amplifies adversarial noise and therefore requires stronger defensive attention, while a low LVS indicates natural robustness at that layer.

---
## Acknowledgements
 
The authors acknowledge the assistance of AI language models (OpenAI's ChatGPT and Anthropic's Claude) for code debugging, documentation refinement, and manuscript preparation. All AI-assisted content was validated and verified by the authors. The research design, experimental implementation, and scientific contributions are entirely the work of the authors.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
<div align="center">
Made with ❤️ for Cybersecurity Research
</div>
