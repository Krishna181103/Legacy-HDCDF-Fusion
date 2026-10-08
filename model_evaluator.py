from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from evaluate import get_labels, plot_confusion_matrix, plot_roc_auc


def evaluate_model(model, X_val, Y_cls_val, class_names, model_name="Model"):

    preds = model.predict(X_val)

    if isinstance(preds, (list, tuple)):
        pred_cls = preds[-1]
    else:
        pred_cls = preds

    y_true, y_pred = get_labels(Y_cls_val, pred_cls)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print(f"\n==============================")
    print(f"📊 {model_name} Evaluation")
    print(f"==============================")
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Macro:", f1_macro)
    print("F1 Micro:", f1_micro)
    print("F1 Weighted:", f1_weighted)

    plot_confusion_matrix(y_true, y_pred, class_names)
    roc_auc, macro_auc = plot_roc_auc(Y_cls_val, pred_cls, class_names)

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1_Macro": f1_macro,
        "F1_Micro": f1_micro,
        "F1_Weighted": f1_weighted,
        "ROC_AUC": macro_auc
    }
