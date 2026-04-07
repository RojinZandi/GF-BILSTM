# GF-BiLSTM: CSI Phase-Aware Deep Fusion for Robotic Activity Recognition

Official implementation of *"Beyond Amplitude: Channel State Information Phase-Aware Deep Fusion for Robotic Activity Recognition"* (ICASSP 2026).

## Overview

GF-BiLSTM is a two-stream gated-fusion BiLSTM that encodes Wi-Fi CSI amplitude and phase separately before adaptively integrating per-time features through a learned gating mechanism. It is evaluated on the [RoboFiSense](https://github.com/siamilab/RoboFiSense) dataset under a Leave-One-Velocity-Out (LOVO) protocol.

## Project Structure

```
├── README.md
├── requirements.txt
├── train.py              # Training script (argparse CLI)
├── evaluate.py           # Evaluation script
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   └── preprocessing.py   # CSI loading, phase sanitization, dataloaders
    ├── models/
    │   ├── __init__.py
    │   └── gf_bilstm.py       # Model + fusion blocks
    └── utils/
        ├── __init__.py
        └── visualization.py   # Confusion matrix plotting
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Amplitude + unwrapped phase (configuration 3 in the paper)
python train.py \
    --data_dir /path/to/8_Class \
    --train_speeds V1 V2 \
    --test_speed V3 \
    --fusion gated \
    --sanitize_phase none \
    --epochs 60

# Amplitude + sanitized phase (configuration 4)
python train.py \
    --data_dir /path/to/8_Class \
    --train_speeds V1 V2 \
    --test_speed V3 \
    --fusion gated \
    --sanitize_phase linear \
    --epochs 60
```

### Evaluation

```bash
python evaluate.py \
    --data_dir /path/to/8_Class \
    --train_speeds V1 V2 \
    --test_speed V3 \
    --checkpoint checkpoints/best_V1_V2_testV3_gated_none.pt \
    --fusion gated \
    --sanitize_phase none \
    --save_cm results/cm.png
```

### Key Arguments

| Argument | Description |
|---|---|
| `--train_speeds` | LOVO training velocities (e.g., `V1 V2`) |
| `--test_speed` | LOVO held-out test velocity (e.g., `V3`) |
| `--fusion` | Fusion strategy: `gated`, `concat`, `hadamard`, `crossattn`, `film` |
| `--sanitize_phase` | Phase preprocessing: `none` (unwrap only), `demean`, `linear` (per-packet LSQ) |
| `--modality_dropout` | Stream-level dropout probability (default 0.05) |

## Citation

```bibtex
@inproceedings{zandi2026beyond,
  title={Beyond Amplitude: Channel State Information Phase-Aware Deep Fusion for Robotic Activity Recognition},
  author={Zandi, Rojin and Salehinejad, Hojjat and Siami, Milad},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```
