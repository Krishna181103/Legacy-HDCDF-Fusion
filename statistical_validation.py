from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evaluation" / "results"
FIG_DIR = ROOT / "paper" / "figures"

CLASS_NAMES = ["Heerup", "Kvium", "Rembrandt", "Sheriff"]

# Reconstructed from the stored BEST_HSI_HYBRID_CENTER confusion-matrix figure.
# Rows are true classes and columns are predicted classes.
BEST_HYBRID_CENTER_CM = np.array(
    [
        [72, 6, 3, 1],
        [5, 56, 6, 3],
        [4, 7, 73, 2],
        [1, 0, 1, 78],
    ],
    dtype=int,
)


def reconstruct_pairs(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = []
    y_pred = []
    for true_idx in range(cm.shape[0]):
        for pred_idx in range(cm.shape[1]):
            count = int(cm[true_idx, pred_idx])
            y_true.extend([true_idx] * count)
            y_pred.extend([pred_idx] * count)
    return np.array(y_true, dtype=int), np.array(y_pred, dtype=int)


def macro_scores(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    precision_values = []
    recall_values = []
    f1_values = []

    for cls_idx in range(n_classes):
        tp = int(np.sum((y_true == cls_idx) & (y_pred == cls_idx)))
        fp = int(np.sum((y_true != cls_idx) & (y_pred == cls_idx)))
        fn = int(np.sum((y_true == cls_idx) & (y_pred != cls_idx)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metrics = {"accuracy": [], "macro_precision": [], "macro_recall": [], "macro_f1": []}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores = macro_scores(y_true[idx], y_pred[idx], n_classes)
        for key, value in scores.items():
            metrics[key].append(value)

    summary = {}
    point = macro_scores(y_true, y_pred, n_classes)
    for key, values in metrics.items():
        arr = np.array(values, dtype=float)
        summary[key] = {
            "point": point[key],
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }
    return summary


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return center - half, center + half


def write_outputs(cm: np.ndarray, bootstrap: dict[str, dict[str, float]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cm_path = RESULTS_DIR / "BEST_HSI_HYBRID_CENTER_confusion_matrix.csv"
    with cm_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, cm):
            writer.writerow([class_name, *row.tolist()])

    summary_path = RESULTS_DIR / "BEST_HSI_HYBRID_CENTER_bootstrap_ci.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "point", "ci95_low", "ci95_high"])
        for metric, values in bootstrap.items():
            writer.writerow([metric, values["point"], values["ci95_low"], values["ci95_high"]])

    successes = int(np.trace(cm))
    total = int(np.sum(cm))
    wilson_low, wilson_high = wilson_interval(successes, total)
    report = {
        "model": "BEST_HSI_HYBRID_CENTER",
        "sample_count": total,
        "correct_count": successes,
        "bootstrap_iterations": 10000,
        "bootstrap_seed": 42,
        "bootstrap_confidence_intervals": bootstrap,
        "wilson_accuracy_ci95": {
            "point": successes / total,
            "ci95_low": wilson_low,
            "ci95_high": wilson_high,
        },
        "mcnemar_test": {
            "status": "not_computed",
            "reason": "Paired per-sample predictions for the baseline models are not stored in the repository.",
        },
        "pca_explained_variance": {
            "status": "not_computed",
            "reason": "The saved PCA pickle requires the original scikit-learn class at load time; scikit-learn is not available in the current runtime.",
        },
    }
    (RESULTS_DIR / "BEST_HSI_HYBRID_CENTER_statistical_validation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    draw_ci_figure(bootstrap, FIG_DIR / "fig_best_hybrid_bootstrap_ci.png")


def draw_ci_figure(bootstrap: dict[str, dict[str, float]], path: Path) -> None:
    labels = ["Accuracy", "Macro precision", "Macro recall", "Macro F1"]
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    width, height = 1500, 900
    margin_left, margin_right = 160, 80
    margin_top, margin_bottom = 120, 170
    y_min, y_max = 0.72, 0.95

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 34)
    small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
    title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 42)

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def x_pos(idx: int) -> float:
        return margin_left + (idx + 0.5) * plot_w / len(keys)

    def y_pos(value: float) -> float:
        return margin_top + (y_max - value) / (y_max - y_min) * plot_h

    draw.text(
        (width / 2, 55),
        "BEST_HSI_HYBRID_CENTER Bootstrap 95% Confidence Intervals",
        fill=(20, 20, 20),
        font=title,
        anchor="mm",
    )
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=(60, 60, 60), width=3)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill=(60, 60, 60), width=3)

    for tick in np.linspace(y_min, y_max, 6):
        y = y_pos(float(tick))
        draw.line((margin_left, y, margin_left + plot_w, y), fill=(225, 225, 225), width=2)
        draw.text((margin_left - 18, y), f"{tick:.2f}", fill=(35, 35, 35), font=small, anchor="rm")

    for idx, (label, key) in enumerate(zip(labels, keys)):
        x = x_pos(idx)
        values = bootstrap[key]
        y_point = y_pos(values["point"])
        y_low = y_pos(values["ci95_low"])
        y_high = y_pos(values["ci95_high"])
        draw.line((x, y_low, x, y_high), fill=(31, 95, 139), width=7)
        draw.line((x - 28, y_low, x + 28, y_low), fill=(31, 95, 139), width=5)
        draw.line((x - 28, y_high, x + 28, y_high), fill=(31, 95, 139), width=5)
        draw.ellipse((x - 18, y_point - 18, x + 18, y_point + 18), fill=(31, 95, 139))
        draw.text((x, margin_top + plot_h + 44), label, fill=(25, 25, 25), font=small, anchor="mm")
        draw.text((x, margin_top + plot_h + 85), f"{values['point']:.4f}", fill=(25, 25, 25), font=small, anchor="mm")

    draw.text((45, margin_top + plot_h / 2), "Score", fill=(25, 25, 25), font=font, anchor="mm")
    img.save(path)


def main() -> None:
    y_true, y_pred = reconstruct_pairs(BEST_HYBRID_CENTER_CM)
    bootstrap = bootstrap_ci(y_true, y_pred, len(CLASS_NAMES))
    write_outputs(BEST_HYBRID_CENTER_CM, bootstrap)
    for metric, values in bootstrap.items():
        print(
            f"{metric}: {values['point']:.4f} "
            f"(95% bootstrap CI {values['ci95_low']:.4f}-{values['ci95_high']:.4f})"
        )


if __name__ == "__main__":
    main()
