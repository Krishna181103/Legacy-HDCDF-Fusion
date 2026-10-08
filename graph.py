import matplotlib.pyplot as plt
import numpy as np

models = ["2D CNN", "3D CNN", "Patch ConvNeXt", "ViT3D"]
accuracy = [58, 80, 76, 67]
reg_mae = [0.60, 0.45, 0.47, 0.48]

x = np.arange(len(models))

plt.figure(figsize=(8,5))
plt.plot(models, accuracy, marker='o', label="Accuracy (%)")
plt.title("Model Comparison - Classification Accuracy")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.savefig("accuracy_graph.png")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(models, reg_mae, marker='o', color='red', label="Regression MAE")
plt.title("Model Comparison - Regression MAE")
plt.ylabel("MAE")
plt.grid(True)
plt.savefig("mae_graph.png")
plt.show()
