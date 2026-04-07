"""Plotting utilities (confusion matrix, etc.)."""

import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    cm: np.ndarray,
    target_names: list[str],
    cmap=None,
    normalize: bool = True,
    save_path: str | None = None,
):
    if cmap is None:
        cmap = plt.get_cmap("Blues")

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=20)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = cm[i, j] * 100 if normalize else cm[i, j]
        fmt = "{:0.2f}" if normalize else "{:,}"
        plt.text(
            j, i, fmt.format(val),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylim(len(target_names) - 0.5, -0.5)
    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
