"""Classification-map utilities for hyperspectral experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_confusion_matrix_heatmap(y_true, y_pred, class_names, out_path, title="Confusion Matrix"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return cm


def save_scene_classification_maps(
    ground_truth_map,
    prediction_map,
    class_names,
    out_dir,
    background_label=-1,
):
    """Save ground-truth, prediction, and error maps for pixel-level HSI scenes.

    This is intended for datasets like Indian Pines/Pavia/Salinas where each
    spatial pixel has a label. HyperLeaf2024 in this repository is image-level
    cultivar classification, so same-dimension scene maps are only meaningful if
    pixel-level labels are added later.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = np.asarray(ground_truth_map)
    pred = np.asarray(prediction_map)
    if gt.shape != pred.shape:
        raise ValueError("ground_truth_map and prediction_map must have the same spatial shape.")

    valid = gt != background_label
    error = np.zeros_like(gt, dtype=np.int32)
    error[valid & (gt != pred)] = 1

    cmap = plt.get_cmap("tab20", len(class_names))
    for arr, name, title in [
        (gt, "ground_truth_map.png", "Ground Truth"),
        (pred, "predicted_classification_map.png", "Prediction"),
    ]:
        plt.figure(figsize=(7, 6))
        plt.imshow(np.ma.masked_where(arr == background_label, arr), cmap=cmap, vmin=0, vmax=len(class_names) - 1)
        plt.colorbar(ticks=np.arange(len(class_names)))
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / name)
        plt.close()

    plt.figure(figsize=(7, 6))
    plt.imshow(np.ma.masked_where(~valid, error), cmap="Reds", vmin=0, vmax=1)
    plt.title("Error Map")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "error_map.png")
    plt.close()


def save_image_level_prediction_grid(y_true, y_pred, class_names, out_path, columns=32):
    """Visualize image-level predictions as a compact sample grid.

    This is not a pixel-level classification map; it is useful for HyperLeaf2024,
    where each TIFF has one cultivar label.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = int(np.ceil(len(y_true) / float(columns)))
    grid = np.full((rows, columns), -1, dtype=np.int32)
    correct = np.zeros((rows, columns), dtype=np.int32)
    for idx, pred in enumerate(y_pred):
        r, c = divmod(idx, columns)
        grid[r, c] = pred
        correct[r, c] = int(pred == y_true[idx])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, max(3, rows * 0.35)))
    plt.imshow(np.ma.masked_where(grid < 0, correct), cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("Image-Level Prediction Correctness Grid")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
