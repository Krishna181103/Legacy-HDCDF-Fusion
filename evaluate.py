import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, roc_curve, auc


def get_labels(y_true, y_pred_prob):
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    if len(class_names) != cm.shape[0]:
        class_names = [f"Class_{i}" for i in range(cm.shape[0])]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    return cm


def compute_f1(y_true, y_pred):
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print("\n===== Classification Report =====\n")
    print(classification_report(y_true, y_pred))

    print("F1 Macro:", f1_macro)
    print("F1 Micro:", f1_micro)
    print("F1 Weighted:", f1_weighted)

    return f1_macro, f1_micro, f1_weighted


def plot_roc_auc(y_true_onehot, y_pred_prob, class_names):

    if len(y_true_onehot.shape) == 1:
        y_true_onehot = np.eye(len(class_names))[y_true_onehot]

    n_classes = y_true_onehot.shape[1]

    if len(class_names) != n_classes:
        class_names = [f"Class_{i}" for i in range(n_classes)]

    fpr, tpr, roc_auc = {}, {}, {}

    plt.figure(figsize=(7, 6))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Multi-class)")
    plt.legend()
    plt.show()

    macro_auc = roc_auc_score(y_true_onehot, y_pred_prob, average="macro", multi_class="ovr")
    print("Macro ROC-AUC:", macro_auc)

    return roc_auc, macro_auc
