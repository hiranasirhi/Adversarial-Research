# LARAR: Layer-wise Adversarial Robustness using Adaptive Regularization

LARAR is a deep learning-based adversarial defense framework for Network Intrusion Detection Systems (NIDS).

The framework improves model robustness by analyzing the vulnerability of individual neural network layers and applying adaptive regularization where it is most needed.

## Key Features

* **Layer Vulnerability Score (LVS):** Measures the sensitivity of hidden-layer representations to adversarial perturbations.
* **Adaptive Regularization:** Adjusts defensive regularization applied to different layers based on their vulnerability.
* **Auxiliary Classifiers:** Provides additional supervision at intermediate layers to improve adversarial detection.
* **Adversarial Training:** Trains the model using both clean and adversarial network traffic.
* **Interpretable Analysis:** Tracks layer-wise vulnerability to provide insight into model robustness.

## Dataset

The project uses the **UNSW-NB15** dataset for network intrusion detection, with traffic classified as normal or attack.

Dataset: https://research.unsw.edu.au/projects/unsw-nb15-dataset

## General Workflow

```text
Network Traffic
       ↓
Data Preprocessing
       ↓
Neural Network
       ↓
Layer Vulnerability Analysis
       ↓
Adaptive Regularization
       ↓
Adversarial Training
       ↓
Robust NIDS Model
```

## Technologies

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

## Project Structure

```text
Adversarial-Robustness-Research/
├── train.py
├── evaluate.py
├── hybrid.py
├── checkpoints/
├── data/
└── README.md
```

## Research Goal

The main goal of LARAR is to develop a more robust and interpretable NIDS by identifying vulnerable neural network layers and dynamically strengthening their resistance to adversarial attacks.

## Acknowledgements

AI language models were used for code debugging, documentation refinement, and manuscript preparation. All AI-assisted content was reviewed and verified by the authors.
