#!/usr/bin/env python3
"""
Evaluate a trained GF-BiLSTM checkpoint on the RoboFiSense test set.

Example
-------
    python evaluate.py \
        --data_dir /path/to/8_Class \
        --train_speeds V1 V2 \
        --test_speed V3 \
        --checkpoint checkpoints/best_V1_V2_testV3_gated_none.pt \
        --fusion gated \
        --sanitize_phase none \
        --save_cm results/cm_V1V2_V3.png
"""

import argparse
import numpy as np
import torch

from sklearn.metrics import classification_report, confusion_matrix

from src.data.preprocessing import (
    get_csi_data, build_dataloaders, CLASS_NAMES,
)
from src.models.gf_bilstm import CSI2StreamBiLSTM
from src.utils.visualization import plot_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate GF-BiLSTM")

    # Data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--train_speeds", nargs="+", default=["V1", "V2"])
    p.add_argument("--test_speed", type=str, default="V3")
    p.add_argument("--target_hz", type=int, default=30)
    p.add_argument("--sanitize_phase", type=str, default="none",
                    choices=["none", "demean", "linear"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--val_size", type=float, default=0.2)

    # Model
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to saved .pt state_dict")
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--first_layers", type=int, default=1)
    p.add_argument("--more_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--fusion", type=str, default="gated",
                    choices=["concat", "gated", "hadamard", "crossattn", "film"])

    # Output
    p.add_argument("--save_cm", type=str, default=None,
                    help="Path to save confusion matrix image (optional)")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    (train_csis, train_labels), (test_csis, test_labels) = get_csi_data(
        directory=args.data_dir,
        train_speeds=args.train_speeds,
        test_speed=args.test_speed,
        target_hz=args.target_hz,
        unwrap_phase=True,
        sanitize_phase=args.sanitize_phase,
    )

    _, _, test_loader, S, num_classes = build_dataloaders(
        train_csis, train_labels, test_csis, test_labels,
        val_size=args.val_size,
        batch_size=args.batch_size,
    )

    # ---- Model ----
    model = CSI2StreamBiLSTM(
        S=S,
        num_classes=num_classes,
        hidden_size=args.hidden_size,
        first_layers=args.first_layers,
        more_layers=args.more_layers,
        dropout=args.dropout,
        fusion=args.fusion,
        modality_dropout_p=0.0,
    ).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # ---- Inference ----
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # ---- Report ----
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    cfm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cfm, CLASS_NAMES, save_path=args.save_cm)


if __name__ == "__main__":
    main()
