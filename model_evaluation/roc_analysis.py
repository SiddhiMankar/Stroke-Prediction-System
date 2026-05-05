import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pickle
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 1. Load Preprocessed Data
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'data_processing', 'preprocessed_data.pkl')
print(f"Loading preprocessed data from {data_path}...")
with open(data_path, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']
input_size = len(feature_names)

# 2. Load the trained model
from tensorflow.keras.models import load_model
model_path = os.path.join(BASE_DIR, 'models', 'final', 'stroke_final_model.h5')
print(f"Loading trained model from {model_path}...")
model = load_model(model_path)

# Step 3: Generate Prediction Probabilities
print("\n--- Generating Prediction Probabilities ---")
y_pred_prob = model.predict(X_test, verbose=0).ravel()

# Step 4: Compute ROC Curve
print("Computing ROC Curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Step 5: Compute AUC Score
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC Score: {auc_score:.4f}")

# Step 6: Plot the ROC Curve
plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, color='blue', lw=2, label="ROC Curve (AUC = %.2f)" % auc_score)
plt.plot([0, 1], [0, 1], color='red', linestyle="--", lw=2, label="Random Guessing")  # diagonal line

plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
plt.ylabel("True Positive Rate (Recall/Sensitivity)", fontsize=12)
plt.title("ROC Curve for Stroke Prediction Model", fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

# Save the plot
output_path = os.path.join(BASE_DIR, "results", "plots", "roc_curve.png")
plt.savefig(output_path)
print(f"\nSaved ROC curve to '{output_path}'")
