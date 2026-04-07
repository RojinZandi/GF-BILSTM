"""
GF-BiLSTM: Two-stream gated-fusion BiLSTM for CSI-based activity recognition.

Supports multiple fusion strategies: concat, gated, hadamard, crossattn, film.
"""

import torch
import torch.nn as nn
from typing import Literal

FUSION_TYPES = Literal["concat", "gated", "hadamard", "crossattn", "film"]


# ---------------------------------------------------------------------------
# Fusion blocks
# ---------------------------------------------------------------------------

class GatedFuse(nn.Module):
    """Per-time gated fusion (Eq. 11-12 in the paper)."""
    def __init__(self, d: int):
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(),
            nn.Linear(d, d), nn.Sigmoid(),
        )

    def forward(self, hA, hP):          # [B, T, D] each
        g = self.g(torch.cat([hA, hP], dim=-1))
        return g * hA + (1 - g) * hP


class HadamardFuse(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.mix = nn.Linear(3 * d, d)

    def forward(self, hA, hP):
        return self.mix(torch.cat([hA, hP, hA * hP], dim=-1))


class CrossAttnFuse(nn.Module):
    def __init__(self, d: int, nhead: int = 4):
        super().__init__()
        self.attn_A2P = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.attn_P2A = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.proj = nn.Linear(2 * d, d)

    def forward(self, hA, hP):
        A2P, _ = self.attn_A2P(hA, hP, hP)
        P2A, _ = self.attn_P2A(hP, hA, hA)
        return self.proj(torch.cat([A2P, P2A], dim=-1))


class FiLMFuse(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.cond = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2 * d),
        )
        self.mix = nn.Linear(d, d)

    def forward(self, hA, hP):
        gamma, beta = self.cond(hP).chunk(2, dim=-1)
        return self.mix(gamma * hA + beta)


def make_fuser(kind: FUSION_TYPES, d: int) -> nn.Module:
    if kind == "concat":
        return nn.Linear(2 * d, d)
    if kind == "gated":
        return GatedFuse(d)
    if kind == "hadamard":
        return HadamardFuse(d)
    if kind == "crossattn":
        return CrossAttnFuse(d)
    if kind == "film":
        return FiLMFuse(d)
    raise ValueError(f"Unknown fusion type: {kind}")


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CSI2StreamBiLSTM(nn.Module):
    """Two-stream BiLSTM with adaptive fusion for CSI classification.

    Args:
        S: Number of subcarriers per stream (after pilot removal & sniffer concat).
        num_classes: Number of activity classes.
        hidden_size: Per-direction LSTM hidden width.
        first_layers: BiLSTM layers per branch (before fusion).
        more_layers: BiLSTM layers after fusion.
        dropout: Dropout probability.
        fusion: Fusion strategy name.
        modality_dropout_p: Probability of zeroing a whole stream during training.
    """

    def __init__(
        self,
        S: int,
        num_classes: int,
        hidden_size: int = 128,
        first_layers: int = 1,
        more_layers: int = 2,
        dropout: float = 0.2,
        fusion: FUSION_TYPES = "gated",
        modality_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.S = S
        self.h = hidden_size
        self.moddrop_p = modality_dropout_p
        self.fusion_kind = fusion

        # Per-time layer normalisation for each stream
        self.normA = nn.LayerNorm(S)
        self.normP = nn.LayerNorm(S)

        # 1-layer BiLSTM per branch
        self.lstmA = nn.LSTM(
            S, hidden_size, num_layers=first_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if first_layers > 1 else 0.0,
        )
        self.lstmP = nn.LSTM(
            S, hidden_size, num_layers=first_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if first_layers > 1 else 0.0,
        )

        # Project [B, T, 2h] -> [B, T, h]
        self.projA = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
        )
        self.projP = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
        )

        # Fusion block -> [B, T, h]
        self.fuser = make_fuser(fusion, hidden_size)

        # Deeper BiLSTM after fusion
        self.lstm_more = nn.LSTM(
            hidden_size, hidden_size, num_layers=more_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if more_layers > 1 else 0.0,
        )

        # Temporal average pooling + classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, num_classes),
        )

    # ----- helpers ----------------------------------------------------------

    def _maybe_modality_drop(self, a, p):
        if not self.training or self.moddrop_p <= 0:
            return a, p
        if torch.rand(1, device=a.device) < self.moddrop_p:
            a = torch.zeros_like(a)
        if torch.rand(1, device=a.device) < self.moddrop_p:
            p = torch.zeros_like(p)
        return a, p

    # ----- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, 2, S, T)`` or ``(B, T, 2*S)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        if x.dim() == 4:
            B, C, S, T = x.shape
            assert C == 2 and S == self.S
            amp = x[:, 0].transpose(1, 2).contiguous()   # (B, T, S)
            pha = x[:, 1].transpose(1, 2).contiguous()
        elif x.dim() == 3:
            B, T, _ = x.shape
            amp, pha = x.split(self.S, dim=-1)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # Normalise per time-step
        amp = self.normA(amp)
        pha = self.normP(pha)

        # Optional modality dropout
        amp, pha = self._maybe_modality_drop(amp, pha)

        # First-level feature extraction
        hA, _ = self.lstmA(amp)       # (B, T, 2h)
        hP, _ = self.lstmP(pha)
        hA = self.projA(hA)           # (B, T, h)
        hP = self.projP(hP)

        # Fusion
        if self.fusion_kind == "concat":
            fused = self.fuser(torch.cat([hA, hP], dim=-1))
        else:
            fused = self.fuser(hA, hP)

        # Deeper BiLSTM
        z, _ = self.lstm_more(fused)  # (B, T, 2h)

        # Temporal mean-pool then classify
        z = self.pool(z.transpose(1, 2)).squeeze(-1)  # (B, 2h)
        return self.head(z)
