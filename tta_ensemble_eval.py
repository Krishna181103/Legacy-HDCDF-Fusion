import os
import json
import pickle

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tifffile as tiff
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


ROOT = "/Users/srivallikarnataka/Desktop/UpdatedHyperSpectralLeafImage"
TRAIN_CSV = os.path.join(ROOT, "datasets", "HyperLeaf2024", "train.csv")
IMG_DIR = os.path.join(ROOT, "datasets", "HyperLeaf2024", "images")
MODEL_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CULTIVAR_COLUMNS = ["Heerup", "Kvium", "Rembrandt", "Sheriff"]
SEED = 42


def load_hsi_cube(image_id):
    image_id = str(image_id).split(".")[0].zfill(5)
    path = os.path.join(IMG_DIR, f"{image_id}.tiff")

    cube = tiff.imread(path).astype(np.float32)
    if cube.ndim == 3 and cube.shape[0] < cube.shape[2]:
        cube = np.transpose(cube, (1, 2, 0))
    cube = (cube - cube.min()) / (cube.max() - cube.min() + 1e-8)
    return cube


def resize_hsi(cube, size):
    h, w, b = cube.shape
    out = np.zeros((size[0], size[1], b), dtype=np.float32)
    for i in range(b):
        out[:, :, i] = cv2.resize(cube[:, :, i], size)
    return out


def spectral_normalize(cube):
    mean = np.mean(cube, axis=2, keepdims=True)
    std = np.std(cube, axis=2, keepdims=True) + 1e-8
    return (cube - mean) / std


def predict_3d_tta(model, cube):
    cube64 = resize_hsi(cube, (64, 64))
    views = [
        cube64,
        np.flip(cube64, axis=0),
        np.flip(cube64, axis=1),
        np.flip(np.flip(cube64, axis=0), axis=1),
    ]
    batch = np.stack(views, axis=0)[..., None]
    preds = model.predict(batch, verbose=0)
    if isinstance(preds, list):
        preds = preds[-1]
    return np.mean(preds, axis=0)


def predict_3d_pca_strong_tta(model, pca_model, cube, crop_size=48):
    cube128 = resize_hsi(cube, (128, 128))
    cube128 = spectral_normalize(cube128)

    h, w, b = cube128.shape
    flat = cube128.reshape(-1, b)
    reduced = pca_model.transform(flat).reshape(h, w, 32).astype(np.float32)

    coords = [
        (0, 0),
        (0, w - crop_size),
        (h - crop_size, 0),
        (h - crop_size, w - crop_size),
        ((h - crop_size) // 2, (w - crop_size) // 2),
    ]

    views = []
    for top, left in coords:
        crop = reduced[top:top + crop_size, left:left + crop_size, :]
        views.append(crop)
        views.append(np.flip(crop, axis=1))

    batch = np.stack(views, axis=0)[..., None]
    preds = model.predict(batch, verbose=0)
    if isinstance(preds, list):
        preds = preds[-1]
    return np.mean(preds, axis=0)


def evaluate_predictions(y_true_onehot, probs, model_name):
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    try:
        roc_auc = roc_auc_score(y_true_onehot, probs, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    print("\n==============================")
    print(f"Evaluating {model_name}")
    print("==============================")
    print("Accuracy:", acc)
    print("Precision macro:", precision)
    print("Recall macro:", recall)
    print("F1 macro:", f1_macro)
    print("F1 micro:", f1_micro)
    print("F1 weighted:", f1_weighted)
    print("ROC-AUC:", roc_auc)
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(CULTIVAR_COLUMNS)),
            target_names=CULTIVAR_COLUMNS,
            zero_division=0,
        )
    )

    return {
        "Model": model_name,
        "Accuracy": float(acc),
        "Precision_macro": float(precision),
        "Recall_macro": float(recall),
        "F1_macro": float(f1_macro),
        "F1_micro": float(f1_micro),
        "F1_weighted": float(f1_weighted),
        "ROC_AUC": float(roc_auc) if not np.isnan(roc_auc) else None,
    }, y_pred


def save_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CULTIVAR_COLUMNS)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CULTIVAR_COLUMNS,
        yticklabels=CULTIVAR_COLUMNS,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} Confusion Matrix")
    path = os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    return path


def search_best_weight(y_true_onehot, probs_a, probs_b, step=0.02):
    y_true = np.argmax(y_true_onehot, axis=1)
    best = None

    weights = np.arange(0.0, 1.0 + 1e-9, step)
    for w in weights:
        probs = w * probs_a + (1.0 - w) * probs_b
        y_pred = np.argmax(probs, axis=1)
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")

        if best is None or acc > best["Accuracy"] or (
            np.isclose(acc, best["Accuracy"]) and f1m > best["F1_macro"]
        ):
            best = {
                "weight_model_a": float(w),
                "weight_model_b": float(1.0 - w),
                "Accuracy": float(acc),
                "F1_macro": float(f1m),
            }

    return best


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    df = pd.read_csv(TRAIN_CSV)
    y_onehot_all = df[CULTIVAR_COLUMNS].values.astype(np.float32)
    df["label_idx"] = np.argmax(y_onehot_all, axis=1)

    _, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label_idx"],
    )
    val_df = val_df.reset_index(drop=True)
    y_true_onehot = val_df[CULTIVAR_COLUMNS].values.astype(np.float32)
    y_true = np.argmax(y_true_onehot, axis=1)

    model_3d_path = os.path.join(MODEL_DIR, "model_3d.keras")
    model_pca_strong_path = os.path.join(MODEL_DIR, "model_3d_pca_strong_best.keras")
    pca_path = os.path.join(MODEL_DIR, "pca32_global.pkl")

    if not os.path.exists(model_3d_path):
        raise FileNotFoundError(f"Missing model: {model_3d_path}")
    if not os.path.exists(model_pca_strong_path):
        raise FileNotFoundError(f"Missing model: {model_pca_strong_path}")
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"Missing PCA model: {pca_path}")

    print("Loading models...")
    model_3d = tf.keras.models.load_model(model_3d_path, compile=False)
    model_pca_strong = tf.keras.models.load_model(model_pca_strong_path, compile=False)
    with open(pca_path, "rb") as f:
        pca_model = pickle.load(f)

    probs_3d = []
    probs_pca_strong = []

    print("Running TTA inference on validation split...")
    total = len(val_df)
    for i, row in val_df.iterrows():
        cube = load_hsi_cube(row["ImageId"])

        p3d = predict_3d_tta(model_3d, cube)
        ppca = predict_3d_pca_strong_tta(model_pca_strong, pca_model, cube, crop_size=48)

        probs_3d.append(p3d)
        probs_pca_strong.append(ppca)

        if (i + 1) % 25 == 0 or (i + 1) == total:
            print(f"Processed {i + 1}/{total}")

    probs_3d = np.array(probs_3d, dtype=np.float32)
    probs_pca_strong = np.array(probs_pca_strong, dtype=np.float32)

    results = []

    metrics_3d, pred_3d = evaluate_predictions(y_true_onehot, probs_3d, "3D_CNN_TTA")
    results.append(metrics_3d)
    cm_path_3d = save_confusion(y_true, pred_3d, "3D_CNN_TTA")

    metrics_pca, pred_pca = evaluate_predictions(y_true_onehot, probs_pca_strong, "3D_CNN_PCA_STRONG_TTA")
    results.append(metrics_pca)
    cm_path_pca = save_confusion(y_true, pred_pca, "3D_CNN_PCA_STRONG_TTA")

    best = search_best_weight(y_true_onehot, probs_3d, probs_pca_strong, step=0.02)
    w3d = best["weight_model_a"]
    wpca = best["weight_model_b"]
    probs_ens = w3d * probs_3d + wpca * probs_pca_strong

    print(f"\nBest ensemble weights: 3D_CNN_TTA={w3d:.2f}, 3D_CNN_PCA_STRONG_TTA={wpca:.2f}")
    metrics_ens, pred_ens = evaluate_predictions(y_true_onehot, probs_ens, "3D_ENSEMBLE_TTA")
    metrics_ens["Weight_3D_CNN_TTA"] = w3d
    metrics_ens["Weight_3D_CNN_PCA_STRONG_TTA"] = wpca
    results.append(metrics_ens)
    cm_path_ens = save_confusion(y_true, pred_ens, "3D_ENSEMBLE_TTA")

    results_df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_DIR, "tta_ensemble_metrics.csv")
    out_json = os.path.join(RESULTS_DIR, "tta_ensemble_metrics.json")
    results_df.to_csv(out_csv, index=False)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    pred_df = pd.DataFrame({
        "ImageId": val_df["ImageId"].values,
        "TrueLabel": y_true,
        "Pred_3D_CNN_TTA": pred_3d,
        "Pred_3D_CNN_PCA_STRONG_TTA": pred_pca,
        "Pred_3D_ENSEMBLE_TTA": pred_ens,
    })
    pred_out = os.path.join(RESULTS_DIR, "tta_ensemble_val_predictions.csv")
    pred_df.to_csv(pred_out, index=False)

    weights_out = os.path.join(RESULTS_DIR, "tta_ensemble_best_weights.json")
    with open(weights_out, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\nSaved files:")
    print(f"- {out_csv}")
    print(f"- {out_json}")
    print(f"- {pred_out}")
    print(f"- {weights_out}")
    print(f"- {cm_path_3d}")
    print(f"- {cm_path_pca}")
    print(f"- {cm_path_ens}")


if __name__ == "__main__":
    main()
