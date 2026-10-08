import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("evaluation/model_comparison.csv")

metrics = ["Accuracy", "F1_Macro", "ROC_AUC"]

for metric in metrics:
    plt.figure(figsize=(8, 5))
    plt.bar(df["Model"], df[metric])
    plt.title(f"Model Comparison - {metric}")
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"evaluation/{metric}_comparison.png")
    plt.show()
