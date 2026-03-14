import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pickle
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 1. Load Preprocessed Data
data_path = 'processed_data/preprocessed_data.pkl'
print(f"Loading preprocessed data from {data_path}...")
with open(data_path, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']
input_size = len(feature_names)

# Rebuild and Train the selected Model (64 -> 32 -> 16)
print("\n--- Rebuilding Selected Model (64 -> 32 -> 16) ---")
model = Sequential([
    Input(shape=(input_size,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Training model...")
model.fit(
    X_train, y_train,
    epochs=30, batch_size=32,
    class_weight=data['class_weights'],
    verbose=0
)

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
plt.savefig("roc_curve.png")
print("\nSaved ROC curve to 'roc_curve.png'")
