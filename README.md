# LARAR: Layer-wise Adversarial Robustness using Adaptive Regularization

Code and experiments for "Enhancing Adversarial Robustness in Network Intrusion Detection: A Layer-wise Adaptive Regularization Approach." LARAR extends adversarial training for NIDS with a Layer Vulnerability Score (LVS), adaptive layer-wise regularization, auxiliary-classifier supervision, and a single-input inference-time detector, evaluated on UNSW-NB15.

## Requirements

```
pip install torch pandas scikit-learn pyarrow matplotlib scipy
```

Python 3.10+. CPU is sufficient; all reported benchmarks use CPU only.

## Data

Place `UNSW_NB15_testing-set.parquet` (Moustafa & Slay, 2015) in the working directory. `data_utils.py` handles preprocessing: 34 features (31 continuous, 3 categorical), 80/20 train-test split with a further 90/10 train/validation split, and a benign-only clean-reference subset used for detector calibration. No test-set data is used at any calibration or training stage.

## Repository structure

| File | Purpose |
|---|---|
| `data_utils.py` | Shared preprocessing pipeline (used by every script below) |
| `phase2_advnn-train.py` | Trains Base ADVNN and Vanilla NN (`--vanilla`, `--matched-capacity` flags) |
| `phase3_adversarial-generation.py` | Evaluates a trained model: clean/FGSM/PGD/transfer accuracy, single-input detector, adaptive detector-aware attack |
| `larar_train.py` | Trains and evaluates LARAR (equivalent scope to phase2+phase3 combined) |
| `run_multiseed_experiments.py` | Orchestrates all three methods across multiple seeds and aggregates results (mean ± std) |
| `ablation_study.py` | Component ablation (LVS regularization, adaptive weighting, auxiliary classifiers), multi-seed |
| `epsilon_sweep.py` | Robustness vs. perturbation budget (ε) across all three methods |
| `loss_coefficient_sensitivity.py` | One-at-a-time sensitivity sweep over the composite loss coefficients |
| `latency_benchmark.py` | Parameter count, FLOPs, inference latency (median/p95/p99), early-exit accuracy-coverage tradeoff |

## Reproducing the results

```bash
# Main results (Tables 3, 5, 6) — 5 seeds
python3 run_multiseed_experiments.py --seeds 42 43 44 45 46

# Re-aggregate without retraining
python3 run_multiseed_experiments.py --seeds 42 43 44 45 46 --skip-training

# Ablation study (Figure 4)
python3 ablation_study.py --seeds 42 43 44 45 46

# Epsilon sweep
python3 epsilon_sweep.py --seeds 42 43 44 45 46

# Loss-coefficient sensitivity
python3 loss_coefficient_sensitivity.py --seed 42

# Latency / FLOPs / early-exit
python3 latency_benchmark.py --model-prefix larar_seed42
```

Each script writes its own `*_results.json` / `*_config.json` alongside training logs, plots (`.png`), and model checkpoints (`.pth`). `multiseed_aggregate_results.json` is the single source for Tables 3, 5, and 6.

## Key implementation notes

- **Layer weights** (`w^(l)`) are computed as an exponential moving average of relative LVS across layers, not learned via gradient descent — this keeps them strictly positive by construction.
- **Inference-time detection** uses a single-input score (median of top-2% per-neuron z-scores against clean reference statistics), distinct from the paired-batch LVS used during training.
- **Adversarial perturbations** are restricted to the 31 continuous features via a fixed mask; categorical features and a non-negativity floor keep generated examples semantically valid.
- **Batch size** is 64 throughout (training and evaluation) for all three methods.

## Citation

If you use this code, please cite the associated paper (see manuscript for full reference).
