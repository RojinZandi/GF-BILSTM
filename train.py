#!/usr/bin/env python3
"""
Train GF-BiLSTM on the RoboFiSense dataset under the LOVO protocol.

Example
-------
    python train.py \
        --data_dir /src/data/classes \
        --train_speeds V1 V2 \
        --test_speed V3 \
        --fusion gated \
        --sanitize_phase none \
        --epochs 60 \
        --batch_size 8 \
        --save_dir checkpoints
"""

import argparse
import os
import logging

import torch
import torch.nn as nn

from src.data.preprocessing import get_csi_data, build_dataloaders
from src.models.gf_bilstm import CSI2StreamBiLSTM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train GF-BiLSTM")

    # Data
    p.add_argument("--data_dir", type=str, required=True,
                    help="Path to the RoboFiSense 8-class directory")
    p.add_argument("--train_speeds", nargs="+", default=["V1", "V2"],
                    help="Speed folders used for training (e.g. V1 V2)")
    p.add_argument("--test_speed", type=str, default="V3",
                    help="Speed folder used for testing")
    p.add_argument("--target_hz", type=int, default=30,
                    help="Target sampling rate (30 = no downsampling)")
    p.add_argument("--sanitize_phase", type=str, default="none",
                    choices=["none", "demean", "linear"],
                    help="Phase sanitization method")

    # Model
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--first_layers", type=int, default=1,
                    help="BiLSTM layers per branch (before fusion)")
    p.add_argument("--more_layers", type=int, default=2,
                    help="BiLSTM layers after fusion")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--fusion", type=str, default="gated",
                    choices=["concat", "gated", "hadamard", "crossattn", "film"])
    p.add_argument("--modality_dropout", type=float, default=0.05,
                    help="Probability of zeroing a whole stream during training")

    # Training
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=2e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--val_size", type=float, default=0.2,
                    help="Fraction of training data used for validation")

    # Output
    p.add_argument("--save_dir", type=str, default="checkpoints",
                    help="Directory for saving model checkpoints")

    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, max_grad_norm, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        running_loss += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Data ----
    logger.info("Loading data from %s  (train=%s, test=%s, sanitize=%s)",
                args.data_dir, args.train_speeds, args.test_speed, args.sanitize_phase)

    (train_csis, train_labels), (test_csis, test_labels) = get_csi_data(
        directory=args.data_dir,
        train_speeds=args.train_speeds,
        test_speed=args.test_speed,
        target_hz=args.target_hz,
        unwrap_phase=True,
        sanitize_phase=args.sanitize_phase,
    )

    train_loader, val_loader, test_loader, S, num_classes = build_dataloaders(
        train_csis, train_labels, test_csis, test_labels,
        val_size=args.val_size,
        batch_size=args.batch_size,
    )

    logger.info("S=%d  num_classes=%d  train=%d  val=%d  test=%d",
                S, num_classes,
                len(train_loader.dataset), len(val_loader.dataset),
                len(test_loader.dataset))

    # ---- Model ----
    model = CSI2StreamBiLSTM(
        S=S,
        num_classes=num_classes,
        hidden_size=args.hidden_size,
        first_layers=args.first_layers,
        more_layers=args.more_layers,
        dropout=args.dropout,
        fusion=args.fusion,
        modality_dropout_p=args.modality_dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # ---- Training loop ----
    best_val_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    tag = f"{'_'.join(args.train_speeds)}_test{args.test_speed}_{args.fusion}_{args.sanitize_phase}"
    best_path = os.path.join(args.save_dir, f"best_{tag}.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, args.max_grad_norm, device,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        logger.info(
            "Epoch %3d/%d | train loss %.4f  acc %.2f%% | val loss %.4f  acc %.2f%%",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            logger.info("  -> Saved best model (val acc %.2f%%)", val_acc)

    # ---- Final test ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info("Test  loss %.4f  acc %.2f%%  (best val acc %.2f%%)",
                test_loss, test_acc, best_val_acc)


if __name__ == "__main__":
    main()
