import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix
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
class_weights = data['class_weights']
feature_names = data['feature_names']
input_size = len(feature_names)

# We selected the 64 -> 32 -> 16 architecture from Phase 2
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
    metrics=[
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

print("Training model...")
model.fit(
    X_train, y_train,
    epochs=30, batch_size=32,
    class_weight=class_weights,
    verbose=0
)

# Step 1: Get Prediction Probabilities
print("\n--- Step 1: Getting Prediction Probabilities ---")
y_probs = model.predict(X_test, verbose=0).flatten()

# Step 2: Define the Thresholds to Test
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print(f"\n--- Step 2: Testing Thresholds: {thresholds} ---")

results = []

for thresh in thresholds:
    # Step 3: Convert Probabilities to Predictions
    y_pred = (y_probs >= thresh).astype(int)
    
    # Step 4: Calculate Evaluation Metrics
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    # Optional: Calculate False Negatives for context
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    results.append({
        "Threshold": thresh,
        "Recall": recall,
        "Precision": precision,
        "False Negatives": fn,
        "True Positives": tp
    })

# Step 5: Create a Comparison Table
results_df = pd.DataFrame(results)

print("\n--- Step 5: Comparison Table ---")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("threshold_optimization_results.csv", index=False)
print("\nSaved results to 'threshold_optimization_results.csv'")

# Visualize the trade-off
plt.figure(figsize=(10, 6))
plt.plot(results_df['Threshold'], results_df['Recall'], marker='o', label='Recall', color='blue')
plt.plot(results_df['Threshold'], results_df['Precision'], marker='x', label='Precision', color='red')
plt.title('Precision-Recall Trade-off across Probability Thresholds')
plt.xlabel('Probability Threshold')
plt.ylabel('Score')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvline(x=0.4, color='green', linestyle='--', label='Optimal Threshold (0.4)')
plt.axvline(x=0.3, color='purple', linestyle='--', label='Optimal Threshold (0.3)')
plt.legend()
plt.tight_layout()
plt.savefig('threshold_tradeoff.png')
print("Saved visualization to 'threshold_tradeoff.png'")
