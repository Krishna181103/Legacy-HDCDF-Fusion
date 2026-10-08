import os
import json
import pickle

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tifffile as tiff


ROOT = "/Users/srivallikarnataka/Desktop/UpdatedHyperSpectralLeafImage"
TRAIN_CSV = os.path.join(ROOT, "datasets", "HyperLeaf2024", "train.csv")
TEST_CSV = os.path.join(ROOT, "datasets", "HyperLeaf2024", "test.csv")
IMG_DIR = os.path.join(ROOT, "datasets", "HyperLeaf2024", "images")
MODEL_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CULTIVAR_COLUMNS = ["Heerup", "Kvium", "Rembrandt", "Sheriff"]
REGRESSION_COLUMNS = ["GrainWeight", "Gsw", "PhiPS2", "Fertilizer"]


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


def main():
    model_3d_path = os.path.join(MODEL_DIR, "model_3d.keras")
    model_pca_strong_path = os.path.join(MODEL_DIR, "model_3d_pca_strong_best.keras")
    pca_path = os.path.join(MODEL_DIR, "pca32_global.pkl")
    weights_path = os.path.join(RESULTS_DIR, "tta_ensemble_best_weights.json")

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

    w3d = 0.5
    wpca = 0.5
    if os.path.exists(weights_path):
        with open(weights_path, "r", encoding="utf-8") as f:
            w = json.load(f)
        w3d = float(w.get("weight_model_a", 0.5))
        wpca = float(w.get("weight_model_b", 0.5))
        print(f"Using tuned weights from {weights_path}: 3D={w3d:.2f}, PCA_STRONG={wpca:.2f}")
    else:
        print("No tuned ensemble weights found; using default 0.5 / 0.5")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    submission = pd.DataFrame({"ImageId": test_df["ImageId"]})
    for col in REGRESSION_COLUMNS:
        submission[col] = float(train_df[col].mean())

    probs = []
    total = len(test_df)
    print("Running ensemble TTA on test set...")
    for i, image_id in enumerate(test_df["ImageId"].values):
        cube = load_hsi_cube(image_id)
        p3d = predict_3d_tta(model_3d, cube)
        ppca = predict_3d_pca_strong_tta(model_pca_strong, pca_model, cube, crop_size=48)
        p = w3d * p3d + wpca * ppca
        probs.append(p)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"Processed {i + 1}/{total}")

    probs = np.array(probs, dtype=np.float32)
    for i, col in enumerate(CULTIVAR_COLUMNS):
        submission[col] = probs[:, i]

    out_path = os.path.join(RESULTS_DIR, "submission_ensemble_tta.csv")
    submission.to_csv(out_path, index=False)
    print(f"Saved submission: {out_path}")


if __name__ == "__main__":
    main()
