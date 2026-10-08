import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split

# =========================================================
# ✅ PROJECT ROOT PATH
# =========================================================
ROOT_DIR = "/Users/srivallikarnataka/Desktop/UpdatedHyperSpectralLeafImage"
sys.path.append(ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "evaluation", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================================
# IMPORT LOADERS
# =========================================================
from utils.data_loader import load_train_val_data, process_dataframe
from utils.data_loader_3d import load_train_val_data_3d

from utils.tiff_loader import load_hyperspectral_tiff
from utils.spectral_ops import resize_hsi, hyperspectral_to_multiband
from utils.spectral_pca import apply_pca, spectral_normalize

from config import TRAIN_CSV, IMG_DIR

# register vit layers
from models import vit3d


def flatten_spectral_channels(x):
    shape = tf.shape(x)
    b, h, w, s, c = shape[0], shape[1], shape[2], shape[3], shape[4]
    return tf.reshape(x, [b, h, w, s * c])


# =========================================================
# LOAD ViT DATA (32×32×16)
# =========================================================
def load_vit_validation_data(val_df):

    X = []
    Y = []

    for _, row in val_df.iterrows():

        image_id = str(row["ImageId"]).split(".")[0].zfill(5)

        path = os.path.join(IMG_DIR, f"{image_id}.tiff")

        hsi = load_hyperspectral_tiff(path)

        hsi = resize_hsi(hsi, size=(32,32))

        multi = hyperspectral_to_multiband(hsi, num_bands=16)

        X.append(multi)

        Y.append(
            row[["Heerup","Kvium","Rembrandt","Sheriff"]]
            .values.astype(np.float32)
        )

    return np.array(X), np.array(Y)


# =========================================================
# LOAD 3D PCA DATA (32×32×32×1)
# =========================================================
def load_3d_pca_validation_data(val_df):

    X = []
    Y = []

    for _, row in val_df.iterrows():

        image_id = str(row["ImageId"]).split(".")[0].zfill(5)
        path = os.path.join(IMG_DIR, f"{image_id}.tiff")

        hsi = load_hyperspectral_tiff(path)
        hsi = resize_hsi(hsi, size=(128, 128))
        hsi = spectral_normalize(hsi)
        hsi = apply_pca(hsi, n_components=32)

        # Match training/train_3d_pca.py preprocessing
        patch = hsi[0:32, 0:32, :]
        X.append(patch[..., None])

        Y.append(
            row[["Heerup", "Kvium", "Rembrandt", "Sheriff"]]
            .values.astype(np.float32)
        )

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# =========================================================
# LOAD 3D PCA FIXED DATA (for strong model)
# =========================================================
def load_3d_pca_fixed_validation_data(val_df, pca_model, crop_size=48):

    X = []
    Y = []

    for _, row in val_df.iterrows():

        image_id = str(row["ImageId"]).split(".")[0].zfill(5)
        path = os.path.join(IMG_DIR, f"{image_id}.tiff")

        hsi = load_hyperspectral_tiff(path)
        hsi = resize_hsi(hsi, size=(128, 128))
        hsi = spectral_normalize(hsi)

        h, w, b = hsi.shape
        flat = hsi.reshape(-1, b)
        reduced = pca_model.transform(flat).reshape(h, w, 32).astype(np.float32)

        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        patch = reduced[top:top + crop_size, left:left + crop_size, :]
        X.append(patch[..., None])

        Y.append(
            row[["Heerup", "Kvium", "Rembrandt", "Sheriff"]]
            .values.astype(np.float32)
        )

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# =========================================================
# LOAD BEST HYBRID DATA (80x80 -> center crop 64, PCA32)
# =========================================================
def load_best_hybrid_validation_data(val_df, pca_model, resize_hw=(64, 64), crop_size=48):

    X = []
    Y = []
    n_components = int(getattr(pca_model, "n_components_", 24))

    for _, row in val_df.iterrows():

        image_id = str(row["ImageId"]).split(".")[0].zfill(5)
        path = os.path.join(IMG_DIR, f"{image_id}.tiff")

        hsi = load_hyperspectral_tiff(path)
        hsi = resize_hsi(hsi, size=resize_hw)
        hsi = spectral_normalize(hsi)

        h, w, b = hsi.shape
        flat = hsi.reshape(-1, b)
        reduced = pca_model.transform(flat).reshape(h, w, n_components).astype(np.float32)

        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        crop = reduced[top:top + crop_size, left:left + crop_size, :]
        X.append(crop[..., None])

        Y.append(
            row[["Heerup", "Kvium", "Rembrandt", "Sheriff"]]
            .values.astype(np.float32)
        )

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# =========================================================
# EVALUATION FUNCTION
# =========================================================
def evaluate_model(model, X, Y_true_onehot, class_names, model_name):

    print("\n==============================")
    print(f"Evaluating {model_name}")
    print("==============================")

    preds = model.predict(X, verbose=0)

    # multi-output model fix
    if isinstance(preds, list):
        preds = preds[1]

    y_true = np.argmax(Y_true_onehot, axis=1)
    y_pred = np.argmax(preds, axis=1)
    pred_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_true)),
            "y_true": y_true,
            "y_pred": y_pred,
            "correct": y_true == y_pred,
        }
    )
    for idx, name in enumerate(class_names):
        pred_df[f"prob_{name}"] = preds[:, idx]
    pred_df.to_csv(
        os.path.join(RESULTS_DIR, f"{model_name}_predictions.csv"),
        index=False,
    )

    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print("Accuracy:", acc)
    print("Precision macro:", precision_macro)
    print("Recall macro:", recall_macro)
    print("F1 macro:", f1_macro)
    print("F1 micro:", f1_micro)
    print("F1 weighted:", f1_weighted)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{model_name}_confusion_matrix.png"
        )
    )

    plt.close()

    try:
        auc_macro = roc_auc_score(
            Y_true_onehot,
            preds,
            multi_class="ovr",
            average="macro"
        )
    except ValueError:
        auc_macro = float("nan")
        print("ROC-AUC: NaN (could not compute because one or more classes were missing in validation labels)")

    if not np.isnan(auc_macro):
        print("ROC-AUC:", auc_macro)

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision_macro": precision_macro,
        "Recall_macro": recall_macro,
        "F1_macro": f1_macro,
        "F1_micro": f1_micro,
        "F1_weighted": f1_weighted,
        "ROC_AUC": auc_macro
    }


# =========================================================
# LOAD DATASETS
# =========================================================

print("Loading datasets...")

# 2D CNN
_, _, _, X_val_2d, _, Y_cls_val_2d, class_map = load_train_val_data()

# ConvNeXt multiband
df = pd.read_csv(TRAIN_CSV)
df["label_idx"] = np.argmax(
    df[["Heerup", "Kvium", "Rembrandt", "Sheriff"]].values,
    axis=1
)
_, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label_idx"]
)
val_df = val_df.reset_index(drop=True)

X_val_patch, _, Y_cls_val_patch = process_dataframe(
    val_df,
    num_bands=16,
    fit_scaler=False,
    use_rgb=False
)

# 3D CNN
_, _, _, X_val_3d, _, Y_cls_val_3d = load_train_val_data_3d()

# ViT
X_val_vit, Y_cls_val_vit = load_vit_validation_data(val_df)

# 3D PCA CNN
X_val_3d_pca, Y_cls_val_3d_pca = load_3d_pca_validation_data(val_df)

# Optional strong 3D PCA CNN inputs
X_val_3d_pca_strong = None
Y_cls_val_3d_pca_strong = None
pca_model_path = os.path.join(ROOT_DIR, "models", "pca32_global.pkl")
if os.path.exists(pca_model_path):
    with open(pca_model_path, "rb") as f:
        pca_model = pickle.load(f)
    X_val_3d_pca_strong, Y_cls_val_3d_pca_strong = load_3d_pca_fixed_validation_data(
        val_df, pca_model, crop_size=48
    )

# Optional best hybrid model inputs
X_val_best_hybrid = None
Y_cls_val_best_hybrid = None
best_hybrid_pca_path = os.path.join(ROOT_DIR, "models", "best_hsi_hybrid_pca.pkl")
if os.path.exists(best_hybrid_pca_path):
    with open(best_hybrid_pca_path, "rb") as f:
        best_hybrid_pca = pickle.load(f)
    X_val_best_hybrid, Y_cls_val_best_hybrid = load_best_hybrid_validation_data(
        val_df, best_hybrid_pca, resize_hw=(64, 64), crop_size=48
    )

class_names = list(class_map.keys())


# =========================================================
# LOAD MODELS
# =========================================================

print("\nLoading models...")

model_2d = tf.keras.models.load_model(
    os.path.join(ROOT_DIR, "models/model_2d.h5"),
    compile=False
)

model_patch = tf.keras.models.load_model(
    os.path.join(ROOT_DIR, "models/model_convnext_advanced.keras"),
    compile=False
)

model_3d = tf.keras.models.load_model(
    os.path.join(ROOT_DIR, "models/model_3d.keras"),
    compile=False
)

model_3d_pca = tf.keras.models.load_model(
    os.path.join(ROOT_DIR, "models/model_3d_pca.keras"),
    compile=False
)

model_3d_pca_strong = None
strong_model_path = os.path.join(ROOT_DIR, "models/model_3d_pca_strong_best.keras")
if os.path.exists(strong_model_path):
    model_3d_pca_strong = tf.keras.models.load_model(
        strong_model_path,
        compile=False
    )

model_best_hybrid = None
best_hybrid_model_path = os.path.join(ROOT_DIR, "models", "model_best_hsi_hybrid_best.keras")
if os.path.exists(best_hybrid_model_path):
    model_best_hybrid = tf.keras.models.load_model(
        best_hybrid_model_path,
        compile=False,
        custom_objects={"flatten_spectral_channels": flatten_spectral_channels}
    )

model_vit = tf.keras.models.load_model(
    os.path.join(ROOT_DIR, "models/vit3d.keras"),
    compile=False
)


# =========================================================
# RUN EVALUATION
# =========================================================

results = []

results.append(
    evaluate_model(
        model_2d,
        X_val_2d,
        Y_cls_val_2d,
        class_names,
        "2D_CNN"
    )
)

results.append(
    evaluate_model(
        model_patch,
        X_val_patch,
        Y_cls_val_patch,
        class_names,
        "ConvNeXt"
    )
)

results.append(
    evaluate_model(
        model_3d,
        X_val_3d,
        Y_cls_val_3d,
        class_names,
        "3D_CNN"
    )
)

results.append(
    evaluate_model(
        model_3d_pca,
        X_val_3d_pca,
        Y_cls_val_3d_pca,
        class_names,
        "3D_CNN_PCA"
    )
)

if model_3d_pca_strong is not None and X_val_3d_pca_strong is not None:
    results.append(
        evaluate_model(
            model_3d_pca_strong,
            X_val_3d_pca_strong,
            Y_cls_val_3d_pca_strong,
            class_names,
            "3D_CNN_PCA_STRONG"
        )
    )

if model_best_hybrid is not None and X_val_best_hybrid is not None:
    results.append(
        evaluate_model(
            model_best_hybrid,
            X_val_best_hybrid,
            Y_cls_val_best_hybrid,
            class_names,
            "BEST_HSI_HYBRID"
        )
    )

results.append(
    evaluate_model(
        model_vit,
        X_val_vit,
        Y_cls_val_vit,
        class_names,
        "ViT"
    )
)


# =========================================================
# SAVE TABLE
# =========================================================

df_results = pd.DataFrame(results)

print("\nFinal Comparison:")
print(df_results)

df_results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# =========================================================
# SAVE GRAPH
# =========================================================

plt.figure(figsize=(8,5))

plt.bar(
    df_results["Model"],
    df_results["Accuracy"]
)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "accuracy_comparison.png"
    )
)

plt.close()

print("\nAll results saved in:")
print(RESULTS_DIR)
