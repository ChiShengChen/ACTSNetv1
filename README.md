# ACTSNet — Attentional Convolution Time Series Network

PyTorch reproduction of **ACTSNet (Attentional Convolution Time Series Network)**, originally proposed as part of a 2021 NTU master's thesis on MDD EEG classification for TMS treatment response prediction.

> **Thesis:** 利用深度學習分析靜息態腦電圖以預測經顱磁刺激對於難治型憂鬱症之療效
> **Source:** National Taiwan University, 2021
> **Link:** <https://tdr.lib.ntu.edu.tw/handle/123456789/82206>
> **DOI:** 10.6342/NTU202101201

## Citation

```bibtex
@mastersthesis{actsnet2021ntu,
  title   = {利用深度學習分析靜息態腦電圖以預測經顱磁刺激對於難治型憂鬱症之療效},
  author  = {國立臺灣大學碩士論文},
  school  = {National Taiwan University},
  year    = {2021},
  doi     = {10.6342/NTU202101201},
  url     = {https://tdr.lib.ntu.edu.tw/handle/123456789/82206}
}
```

## Overview

ACTSNet is designed for **MDD (Major Depressive Disorder)** resting-state EEG binary classification — predicting **rTMS/iTBS treatment responders vs non-responders**. It is a modified version of [TapNet](https://arxiv.org/abs/2002.07764) (Zhang et al., AAAI 2020) where the LSTM branch is replaced by an **Attentional Convolution (AC)** module inspired by the [Encoder network](https://arxiv.org/abs/1805.03908) (Serra et al., 2018). The motivation is that LSTM does not perform well on noisy MDD EEG data, while AC provides better feature extraction through attention-weighted convolution.

## Architecture

```
Input (batch, 35, T)
  │
  ├─ Branch 1: MultiScaleEncoder
  │    Random dim permutation → grouped Conv1D+BN+ReLU → GlobalPool → Concat
  │
  ├─ Branch 2: ACEncoder (replaces TapNet's LSTM)
  │    Conv1d 1×1 projection → 3×[Conv1D + InstanceNorm + PReLU]
  │    → Softmax(AC) → ⊙ Mul → FC → Sigmoid + InstanceNorm → GlobalAvgPool
  │
  └─ Concatenate → FC → Prototypical Learning → Softmax(−‖·‖²) → Classification
```

**Key components:**

- **Attentional Convolution (AC):** `Softmax(X) ⊙ X` — softmax attention over channel dimension followed by element-wise multiplication, re-weighting feature representations
- **Multi-Scale Encoding:** Random dimension permutation splits 35 channels into groups; each group passes through shared-parameter Conv1D+BN+ReLU with global pooling
- **Supervised Attentional Prototype Learning:** Per-class learnable attention weights (`w_k`, `V_k`) compute attention-weighted prototypes; classification via softmax over negative squared Euclidean distances

## Input Data

- **Electrodes:** 7 frontal channels — FP1, FP2, F7, F3, Fz, F4, F8
- **Sub-bands:** 5 per electrode — alpha (α), beta (β), gamma (γ), delta (δ), theta (θ)
- **Shape:** `(n_samples, 35, T)` — 35 = 7 electrodes × 5 sub-bands
- **Labels:** 0 = non-responder, 1 = responder

```
data/
├── data.npy      # (n_samples, 35, n_timesteps)
└── labels.npy    # (n_samples,)
```

## Installation

```bash
pip install torch numpy scikit-learn
```

## Training

```bash
python -m actsnet.train \
    --data_dir data/ \
    --output_dir checkpoints/ \
    --epochs 200 \
    --batch_size 16 \
    --lr 1e-3
```

| Parameter | Default | Description |
|---|---|---|
| `--n_groups` | 3 | Number of random dimension permutation groups |
| `--prototype_dim` | 128 | Embedding dimension `d` |
| `--latent_dim_u` | 64 | Latent space dimension `u` for prototype attention |
| `--dropout` | 0.1 | Dropout rate |
| `--seed` | 42 | Random seed |
| `--device` | cuda | `cuda` or `cpu` |

## Evaluation

```bash
python -m actsnet.evaluate \
    --checkpoint checkpoints/best_model.pt \
    --data_dir test_data/ \
    --output results.json
```

Reports accuracy, precision, recall, F1-score, AUC-ROC, confusion matrix, and per-sample predictions.

## Project Structure

```
ACTSNet/
├── README.md
├── actsnet/
│   ├── __init__.py
│   ├── config.py       # ACTSNetConfig dataclass
│   ├── dataset.py      # EEGDataset + augmentation
│   ├── model.py        # Full model architecture
│   ├── train.py        # Training loop
│   └── evaluate.py     # Evaluation & metrics
```

## References

- **TapNet:** Zhang et al., "TapNet: Multivariate Time Series Classification with Attentional Prototypical Network," AAAI 2020
- **Encoder:** Serra et al., "Towards a Universal Neural Network Encoder for Time Series," CCIA 2018
- **Prototypical Networks:** Snell et al., "Prototypical Networks for Few-shot Learning," NeurIPS 2017
