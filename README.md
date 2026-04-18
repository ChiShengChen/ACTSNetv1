# ACTSNet — Attentional Convolution Time Series Network

PyTorch reproduction of **ACTSNet (Attentional Convolution Time Series Network)**, originally proposed as part of a 2021 NTU master's thesis on MDD EEG classification for TMS treatment response prediction.

> **Thesis:** 基於注意力機制之時間序列原型卷積神經網路與傳統及量子機器學習模型應用於重度憂鬱症腦波之經顱磁刺激抗憂鬱療效預測與分析
> (EEG Analysis for Prediction of Antidepressant Responses of Transcranial Magnetic Stimulation in Major Depressive Disorder Based on Attentional Convolution Time Series Prototypical Neural Network Model and Classical/Quantum Machine Learning Approaches)
> **Author:** Chi-Sheng Chen (陳麒升)
> **Source:** National Taiwan University, 2021
> **Link:** <https://tdr.lib.ntu.edu.tw/handle/123456789/82206>
> **DOI:** 10.6342/NTU202101201

## Citation

```bibtex
@mastersthesis{chen2021actsnet,
  title   = {基於注意力機制之時間序列原型卷積神經網路與傳統及量子機器學習模型應用於重度憂鬱症腦波之經顱磁刺激抗憂鬱療效預測與分析},
  author  = {Chen, Chi-Sheng},
  school  = {National Taiwan University},
  year    = {2021},
  doi     = {10.6342/NTU202101201},
  url     = {https://tdr.lib.ntu.edu.tw/handle/123456789/82206}
}
```

## Overview

ACTSNet is designed for **MDD (Major Depressive Disorder)** resting-state EEG binary classification — predicting **rTMS/iTBS treatment responders vs non-responders**. It is a modified version of [TapNet](https://ojs.aaai.org/index.php/AAAI/article/view/6165) (Zhang et al., AAAI 2020) where the LSTM branch is replaced by an **Attentional Convolution (AC)** module inspired by the [Encoder network](https://arxiv.org/abs/1805.03908) (Serra et al., 2018). The motivation is that LSTM does not perform well on noisy MDD EEG data, while AC provides better feature extraction through attention-weighted convolution.

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

## Benchmark Results

ACTSNet evaluated on 4 EEG-FM-Bench datasets with 3 seeds (42, 123, 456), 100 finetune epochs, batch size 64, LR 1e-3, T=1024. Metric: **balanced accuracy** (mean ± std).

### From scratch (no pretrain)

| Dataset | Classes | Bal. Acc. | Cohen's κ | Weighted F1 |
|---|---|---|---|---|
| bcic_2a | 4 | 0.4358 ± 0.0263 | 0.2477 ± 0.0350 | 0.4249 ± 0.0451 |
| tuab    | 2 | 0.7457 ± 0.0019 | 0.4960 ± 0.0055 | 0.7497 ± 0.0026 |
| tuev    | 6 | 0.3868 ± 0.0543 | 0.2889 ± 0.0199 | 0.5421 ± 0.0102 |
| seed_iv | 4 | 0.3008 ± 0.0051 | 0.0647 ± 0.0056 | 0.2764 ± 0.0106 |

### With self-supervised pretrain (v1, small pool)

Pretrain: **seed_iv + bcic_2a** (13,441 samples), 50 epochs, NT-Xent (contrastive) + MSE (reconstruction).

| Dataset | Classes | Bal. Acc. | Cohen's κ | Weighted F1 |
|---|---|---|---|---|
| bcic_2a | 4 | 0.4034 ± 0.0433 | 0.2045 ± 0.0577 | 0.3704 ± 0.0715 |

### With self-supervised pretrain (v2, full pool)

Pretrain: **tuab + tuev + bcic_2a + seed_iv + siena_scalp** (74,141 samples), 50 epochs, NT-Xent (contrastive) + MSE (reconstruction).

| Dataset | Classes | Bal. Acc. | Cohen's κ | Weighted F1 | Δ vs from-scratch |
|---|---|---|---|---|---|
| bcic_2a | 4 | 0.3709 ± 0.0262 | 0.1613 ± 0.0350 | 0.3308 ± 0.0538 | −0.065 |
| tuab    | 2 | 0.7257 ± 0.0019 | 0.4574 ± 0.0041 | 0.7301 ± 0.0020 | −0.020 |
| tuev    | 6 | 0.4130 ± 0.0306 | 0.3557 ± 0.0479 | 0.6088 ± 0.0398 | **+0.026** |
| seed_iv | 4 | 0.3045 ± 0.0096 | 0.0687 ± 0.0137 | 0.2729 ± 0.0201 | +0.004 |

Reproduce with:

```bash
# Pretrain (50 epochs, ~80 min on RTX 3090)
python run_pretrain.py \
    --epochs 50 --batch_size 64 --lr 1e-3 \
    --max_channels 64 --max_time_len 1024 \
    --max_samples_per_dataset 20000 \
    --output_dir checkpoints/pretrain_full

# Finetune benchmark on 4 datasets × 3 seeds
python run_eegfm_benchmark.py \
    --datasets bcic_2a tuab tuev seed_iv \
    --pretrained_path checkpoints/pretrain_full/pretrain_final.pt \
    --seeds 42 123 456 --epochs 100 \
    --output_dir checkpoints/eegfm_benchmark_pretrain_full
```

## References

- **TapNet:** Zhang et al., "TapNet: Multivariate Time Series Classification with Attentional Prototypical Network," AAAI 2020
- **Encoder:** Serra et al., "Towards a Universal Neural Network Encoder for Time Series," CCIA 2018
- **Prototypical Networks:** Snell et al., "Prototypical Networks for Few-shot Learning," NeurIPS 2017
