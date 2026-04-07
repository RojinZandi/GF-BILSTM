# GF-BiLSTM: Beyond Amplitude — CSI Phase-Aware Deep Fusion for Robotic Activity Recognition

[![Paper](https://img.shields.io/badge/ICASSP%202026-Paper-blue)](https://arxiv.org/abs/2603.09047)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of:

> **Beyond Amplitude: Channel State Information Phase-Aware Deep Fusion for Robotic Activity Recognition**  
> Rojin Zandi, Hojjat Salehinejad, and Milad Siami  
> *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026*

---

## Overview

Wi-Fi Channel State Information (CSI) has emerged as a promising sensing modality for robotic activity recognition. However, prior work predominantly relies on CSI **amplitude** while underutilizing **phase** information. We propose **GF-BiLSTM** (GateFusion-BiLSTM), a two-stream gated-fusion network that:

1. Encodes amplitude and phase **separately** via per-stream BiLSTMs  
2. **Adaptively fuses** per-time features through a learned gating mechanism  
3. Achieves **state-of-the-art** accuracy and cross-speed robustness on the RoboFiSense benchmark

### Architecture

<p align="center">
  <img src="Figs/flow.pdf" alt="GF-BiLSTM Architecture" width="90%">
</p>

Complex CSI is split into amplitude and phase. The phase branch performs temporal unwrapping and optional packet-wise linear sanitization, then per-time layer normalization. Each stream is encoded by a 1-layer BiLSTM, their per-time features are fused by a learned gate, and a deeper BiLSTM with a final classifier produces action labels.

---

## Key Results

Evaluated under the **Leave-One-Velocity-Out (LOVO)** protocol on the [RoboFiSense](https://github.com/siamilab/RoboFiSense) dataset (8 robotic arm activities at 3 speeds):

| Train | Test | Model | Phase Only | Amp Only | Amp+Phase (Unw) | Amp+Phase (San) |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| V1&V2 | V3 | CNN | 24.28 | 60.33 | 74.84 | 76.33 |
| | | BiLSTM | 31.96 | 78.26 | 89.22 | 91.11 |
| | | BiVTC | 36.09 | 87.50 | 92.76 | 93.85 |
| | | **GF-BiLSTM** | — | — | 93.21 | **96.11** |
| V1&V3 | V2 | BiVTC | 41.73 | 86.96 | 93.15 | 95.20 |
| | | **GF-BiLSTM** | — | — | 93.78 | **95.65** |
| V2&V3 | V1 | BiVTC | 35.76 | 84.89 | 91.74 | 94.86 |
| | | **GF-BiLSTM** | — | — | 94.33 | **95.10** |

**Takeaways:**
- Phase-only is weakest; amplitude-only is stronger; amplitude+phase is best
- Sanitized phase provides a small but consistent gain over unwrapped phase
- GF-BiLSTM achieves the highest accuracy across all LOVO splits

---

## Project Structure

    .
    ├── README.md
    ├── requirements.txt
    ├── train.py                # Training script (CLI)
    ├── evaluate.py             # Evaluation + confusion matrix
    └── src/
        ├── data/
        │   └── preprocessing.py    # CSI loading, phase unwrapping/sanitization
        ├── models/
        │   └── gf_bilstm.py        # GF-BiLSTM + fusion blocks
        └── utils/
            └── visualization.py     # Confusion matrix plotting

---

## Installation

    pip install -r requirements.txt

Requires Python 3.10+ and PyTorch 2.0+.

---

## Usage

### Training

    # Config 3: Amplitude + Unwrapped Phase
    python train.py \
        --data_dir /path/to/8_Class \
        --train_speeds V1 V2 \
        --test_speed V3 \
        --fusion gated \
        --sanitize_phase none \
        --epochs 60

    # Config 4: Amplitude + Sanitized Phase
    python train.py \
        --data_dir /path/to/8_Class \
        --train_speeds V1 V2 \
        --test_speed V3 \
        --fusion gated \
        --sanitize_phase linear \
        --epochs 60

### Evaluation

    python evaluate.py \
        --data_dir /path/to/8_Class \
        --train_speeds V1 V2 \
        --test_speed V3 \
        --checkpoint checkpoints/best_V1_V2_testV3_gated_none.pt \
        --fusion gated \
        --sanitize_phase none \
        --save_cm results/cm.png

### Full LOVO Protocol

    # Split 1: Train V1+V2, Test V3
    python train.py --data_dir ... --train_speeds V1 V2 --test_speed V3

    # Split 2: Train V1+V3, Test V2
    python train.py --data_dir ... --train_speeds V1 V3 --test_speed V2

    # Split 3: Train V2+V3, Test V1
    python train.py --data_dir ... --train_speeds V2 V3 --test_speed V1

### Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--fusion` | Fusion strategy: `gated`, `concat`, `hadamard`, `crossattn`, `film` | `gated` |
| `--sanitize_phase` | `none` (unwrap only), `demean`, `linear` (per-packet LSQ) | `none` |
| `--modality_dropout` | Stream-level dropout probability | `0.05` |
| `--hidden_size` | Per-direction LSTM hidden width | `128` |
| `--epochs` | Number of training epochs | `60` |
| `--batch_size` | Batch size | `8` |

---

## Method Summary

### Phase Preprocessing

1. **Temporal Unwrapping**: Removes 2π discontinuities along the time axis for each subcarrier
2. **Linear Sanitization** (optional): Per-packet least-squares removal of linear trend across subcarriers, eliminating hardware-induced slope and offset

### Gated Fusion

The fusion gate learns to adaptively weight amplitude vs. phase at each time step:

    g_t = sigmoid(W_g [u_A_t ; u_P_t] + b_g)
    z_t = g_t * u_A_t + (1 - g_t) * u_P_t

This allows the model to **down-weight noisy phase windows** and fall back on amplitude when needed, while exploiting phase when informative.

---

## Dataset

This work uses the [RoboFiSense](https://github.com/siamilab/RoboFiSense) dataset:
- **8 activities**: Arc, Elbow, Rectangle, Silence, SLFW, SLRL, SLUD, Triangle
- **3 velocities**: Low (V1), Medium (V2), High (V3)
- **Franka Emika** robotic arm with Wi-Fi CSI capture

---

## Citation

    @inproceedings{zandi2026beyond,
      title={Beyond Amplitude: Channel State Information Phase-Aware Deep Fusion for Robotic Activity Recognition},
      author={Zandi, Rojin and Salehinejad, Hojjat and Siami, Milad},
      booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      year={2026}
    }

## Acknowledgments

This material is based upon work supported in part by the U.S. Office of Naval Research under Grant Award N00014-21-1-2431; and in part by the U.S. National Science Foundation under Grant Award 2208182.
