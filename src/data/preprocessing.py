"""
CSI data loading, phase preprocessing (unwrapping & sanitization),
downsampling, and train/test splitting for the RoboFiSense dataset.
"""

import os
import pickle
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Broadcom 256-tone pilot/unused subcarrier indices to drop
RM_IDX = [
    0, 1, 2, 3, 4, 115, 116, 117, 118, 119, 120, 122, 123, 124,
    125, 126, 127, 128, 129, 130, 131, 253, 254, 255,
]

CLASS_MAP = {
    "Arc": 0, "Elbow": 1, "Rectangle": 2, "Silence": 3,
    "SLFW": 4, "SLRL": 5, "SLUD": 6, "Triangle": 7,
}

CLASS_NAMES = [
    "Arc", "Elbow", "Rectangle", "Silence",
    "SLFW", "SLRL", "SLUD", "Triangle",
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _remove_pilots(arr: np.ndarray, rm_idx: list[int]) -> np.ndarray:
    """Remove pilot/unused subcarrier columns.  arr: (T, S_full) -> (T, S_kept)"""
    return np.delete(arr, rm_idx, axis=1)


def _linear_sanitize_per_packet(phase_ts: np.ndarray) -> np.ndarray:
    """Per-packet linear de-trending across subcarriers (Eq. 5-6 in the paper).

    For each time index *t*, unwrap the phase across subcarriers, fit
    ``phi_t[k] ~ alpha_t * k + beta_t`` via least squares, and subtract.

    Args:
        phase_ts: (T, S) float – phase with RM subcarriers already removed.

    Returns:
        Sanitized phase of the same shape, float32.
    """
    T, S = phase_ts.shape
    if S < 2:
        return phase_ts.copy()

    k = np.arange(S, dtype=np.float64)
    K = np.vstack([k, np.ones_like(k)]).T          # design matrix [k, 1]
    sanitized = np.empty_like(phase_ts, dtype=np.float64)

    for t in range(T):
        y = np.unwrap(phase_ts[t, :].astype(np.float64))
        a, b = np.linalg.lstsq(K, y, rcond=None)[0]
        sanitized[t, :] = y - (a * k + b)

    return sanitized.astype(np.float32)


def _sanitize_phase_pipeline(
    phase_ts: np.ndarray,
    method: str = "linear",
    unwrap_time: bool = True,
) -> np.ndarray:
    """Apply chosen sanitization then (optionally) unwrap along time.

    Args:
        phase_ts: (T, S) after removing RM subcarriers.
        method: ``'none'`` | ``'demean'`` | ``'linear'``.
        unwrap_time: If True, unwrap along time after sanitization.
    """
    if method == "none":
        out = phase_ts.astype(np.float32)
    elif method == "demean":
        out = (phase_ts - phase_ts.mean(axis=1, keepdims=True)).astype(np.float32)
    elif method == "linear":
        out = _linear_sanitize_per_packet(phase_ts)
    else:
        raise ValueError(f"Unknown sanitization method: {method}")

    if unwrap_time:
        out = np.unwrap(out, axis=0).astype(np.float32)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-stream amplitude / phase extraction
# ---------------------------------------------------------------------------

def amp_phase_per_stream(
    complex_ts: np.ndarray,
    rm_idx: list[int],
    unwrap_phase: bool = True,
    sanitize_phase: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract amplitude and (optionally sanitized) phase from complex CSI.

    Args:
        complex_ts: (T, S_full) complex array.
        rm_idx: Subcarrier indices to drop.
        unwrap_phase: Whether to unwrap along time after sanitization.
        sanitize_phase: ``'none'`` | ``'demean'`` | ``'linear'``.

    Returns:
        ``(amp, phase)`` each of shape ``(T, S_kept)``, float32.
    """
    amp_full = np.abs(complex_ts).astype(np.float32)
    phase_full = np.angle(complex_ts).astype(np.float32)

    amp_kept = np.delete(amp_full, rm_idx, axis=1).astype(np.float32)
    phase_kept = np.delete(phase_full, rm_idx, axis=1).astype(np.float32)

    phase_clean = _sanitize_phase_pipeline(
        phase_kept, method=sanitize_phase, unwrap_time=unwrap_phase,
    )
    return amp_kept, phase_clean


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------

def downsample_data(
    arr: np.ndarray, target_hz: int, original_hz: int = 30,
) -> np.ndarray:
    """Downsample ``arr`` (T, S) from *original_hz* to *target_hz*."""
    if target_hz == original_hz:
        return arr
    ratios = {25: (6, 5), 20: (3, 2), 15: (2, 1), 10: (3, 1), 5: (6, 1)}
    if target_hz not in ratios:
        raise ValueError(f"Unsupported target frequency: {target_hz}")
    every, _ = ratios[target_hz]
    if target_hz == 25:
        idx = [i for i in range(arr.shape[0]) if i % 6 != 5]
    elif target_hz == 20:
        idx = [i for i in range(arr.shape[0]) if i % 3 != 2]
    elif target_hz == 15:
        idx = [i for i in range(arr.shape[0]) if i % 2 == 0]
    elif target_hz == 10:
        idx = [i for i in range(arr.shape[0]) if i % 3 == 0]
    elif target_hz == 5:
        idx = [i for i in range(arr.shape[0]) if i % 6 == 0]
    return arr[idx, :]


# ---------------------------------------------------------------------------
# Main dataset loader
# ---------------------------------------------------------------------------

def get_csi_data(
    directory: str,
    train_speeds: list[str],
    test_speed: str,
    target_hz: int = 30,
    unwrap_phase: bool = True,
    sanitize_phase: str = "none",
) -> tuple:
    """Load and preprocess CSI data under the LOVO protocol.

    Each sample is shaped ``(2, S, T)`` – channel 0 = amplitude,
    channel 1 = phase.  Multiple sniffers are concatenated along S.

    Returns:
        ``(train_csis, train_labels), (test_csis, test_labels)``
    """
    train_csis, train_labels = [], []
    test_csis, test_labels = [], []

    for label_name in sorted(os.listdir(directory)):
        label_path = os.path.join(directory, label_name)
        if not os.path.isdir(label_path) or label_name not in CLASS_MAP:
            continue
        label = CLASS_MAP[label_name]

        for speed_folder in os.listdir(label_path):
            if speed_folder in train_speeds:
                split = "train"
            elif speed_folder == test_speed:
                split = "test"
            else:
                continue

            speed_path = os.path.join(label_path, speed_folder)
            for csi_file in sorted(os.listdir(speed_path)):
                if csi_file.startswith("."):
                    continue
                fp = os.path.join(speed_path, csi_file)
                with open(fp, "rb") as f:
                    csi_list = pickle.load(f)

                amp_streams, phase_streams = [], []
                for entry in csi_list:
                    X = np.asarray(entry["complex_csi"])
                    if target_hz != 30:
                        X = downsample_data(X, target_hz)
                    a, p = amp_phase_per_stream(
                        X, rm_idx=RM_IDX,
                        unwrap_phase=unwrap_phase,
                        sanitize_phase=sanitize_phase,
                    )
                    amp_streams.append(a)
                    phase_streams.append(p)

                amp_all = np.concatenate(amp_streams, axis=1)
                phase_all = np.concatenate(phase_streams, axis=1)

                # (T, S) -> (S, T) then stack -> (2, S, T)
                sample = np.stack(
                    [amp_all.T, phase_all.T], axis=0,
                ).astype(np.float32)

                if split == "train":
                    train_csis.append(sample)
                    train_labels.append(label)
                else:
                    test_csis.append(sample)
                    test_labels.append(label)

    train_csis, train_labels = shuffle(
        train_csis, train_labels, random_state=817328462,
    )
    test_csis, test_labels = shuffle(
        test_csis, test_labels, random_state=817328462,
    )
    return (train_csis, train_labels), (test_csis, test_labels)


def build_dataloaders(
    train_csis, train_labels, test_csis, test_labels,
    val_size: float = 0.2,
    batch_size: int = 8,
):
    """Convert lists to tensors, split a validation set, return DataLoaders."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    train_csis = np.array(train_csis)
    train_labels = np.array(train_labels)
    test_csis = np.array(test_csis)
    test_labels = np.array(test_labels)

    X_train, X_val, y_train, y_val = train_test_split(
        train_csis, train_labels, test_size=val_size, shuffle=True,
        random_state=42,
    )

    def _to_loader(X, y, shuffle_flag, bs):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=bs, shuffle=shuffle_flag, drop_last=False)

    train_loader = _to_loader(X_train, y_train, True, batch_size)
    val_loader = _to_loader(X_val, y_val, False, batch_size)
    test_loader = _to_loader(test_csis, test_labels, False, batch_size)

    S = X_train.shape[2]
    num_classes = len(set(y_train))
    return train_loader, val_loader, test_loader, S, num_classes