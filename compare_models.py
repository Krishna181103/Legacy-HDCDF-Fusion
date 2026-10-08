import sys, os
import tensorflow as tf
import pandas as pd

# ================================
# ✅ Add project root to path
# ================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# ================================
# ✅ Import data loaders
# ================================
from utils.data_loader import load_train_val_data              # 2D CNN (RGB-like)
from utils.data_loader_patch import load_train_val_data_patch  # Patch CNN (16-band)
from utils.data_loader_3d import load_train_val_data_3d        # 3D CNN & ViT (HSI cube)

# ================================
# ✅ Import evaluator
# ================================
from evaluation.model_evaluator import evaluate_model


print("\n==============================")
print("🚀 Loading Validation Data...")
print("==============================")

# ================================
# 1️⃣ 2D CNN DATA (3 channels)
# ================================
X_train_2d, Y_reg_train_2d, Y_cls_train_2d, X_val_2d, Y_reg_val_2d, Y_cls_val_2d, class_map = load_train_val_data()
class_names = list(class_map.keys())

# ================================
# 2️⃣ PATCH CNN DATA (16 bands)
# ================================
X_train_p, Y_reg_train_p, Y_cls_train_p, X_val_p, Y_reg_val_p, Y_cls_val_p, class_map_p = load_train_val_data_patch()

# ================================
# 3️⃣ 3D CNN & ViT DATA (HSI cube)
# ================================
X_train_3d, Y_reg_train_3d, Y_cls_train_3d, X_val_3d, Y_reg_val_3d, Y_cls_val_3d = load_train_val_data_3d()
class_names_3d = ["Heerup", "Kvium", "Rembrandt", "Sheriff"]


print("\n==============================")
print("🤖 Loading Models...")
print("==============================")

# ================================
# ✅ Load Models
# ================================
model_2d = tf.keras.models.load_model("models/model_2d.h5", compile=False)
model_patch = tf.keras.models.load_model("models/model_convnext_advanced.keras", compile=False)
model_3d = tf.keras.models.load_model("models/model_3d.keras", compile=False)
model_vit = tf.keras.models.load_model("models/vit3d.keras", compile=False)


print("\n==============================")
print("📊 Evaluating Models...")
print("==============================")

results = []

# ================================
# ✅ Evaluate 2D CNN
# ================================
results.append(
    evaluate_model(model_2d, X_val_2d, Y_cls_val_2d, class_names, "2D Multiband CNN (RGB-like)")
)

# ================================
# ✅ Evaluate Patch ConvNeXt CNN
# ================================
results.append(
    evaluate_model(model_patch, X_val_p, Y_cls_val_p, class_names, "ConvNeXt Patch CNN (16-band)")
)

# ================================
# ✅ Evaluate 3D CNN
# ================================
results.append(
    evaluate_model(model_3d, X_val_3d, Y_cls_val_3d, class_names_3d, "3D CNN (HSI Cube)")
)

# ================================
# ✅ Evaluate ViT
# ================================
results.append(
    evaluate_model(model_vit, X_val_3d, Y_cls_val_3d, class_names_3d, "ViT (HSI Cube)")
)

# ================================
# ✅ Save Comparison Table
# ================================
df_results = pd.DataFrame(results)

print("\n==============================")
print("🏆 MODEL COMPARISON TABLE")
print("==============================")
print(df_results)

df_results.to_csv("evaluation/model_comparison.csv", index=False)

print("\n✅ Comparison results saved to: evaluation/model_comparison.csv")
