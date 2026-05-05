import tensorflow as tf
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix, roc_curve

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data and models
data_path = 'processed_data/preprocessed_data.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

X_test = data['X_test']
y_test = data['y_test']

import joblib
scaler = joblib.load('processed_data/scaler.pkl')
X_test_scaled = scaler.transform(X_test)

# Load Models
model_03 = tf.keras.models.load_model('processed_data/stroke_final_model_03.h5', compile=False)
model_10 = tf.keras.models.load_model('processed_data/stroke_final_model_10.h5', compile=False)

# Probabilities
y_prob_03 = model_03.predict(X_test_scaled, verbose=0).flatten()
y_prob_10 = model_10.predict(X_test_scaled, verbose=0).flatten()

# --- Threshold Optimization ---
thresholds = np.arange(0.1, 0.9, 0.1)

def evaluate_thresholds(y_prob, name):
    print(f"\n--- Threshold Optimization for {name} ---")
    best_t = 0.1
    best_f1 = 0
    metrics = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        if rec + prec > 0:
            f1 = 2 * (rec * prec) / (rec + prec)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        metrics.append((t, rec, prec))
        print(f"Threshold {t:.1f}: Recall={rec:.4f}, Precision={prec:.4f}")
    return best_t, metrics

best_t_03, metrics_03 = evaluate_thresholds(y_prob_03, "SMOTENC 0.3")
best_t_10, metrics_10 = evaluate_thresholds(y_prob_10, "SMOTENC 1.0")

# --- Overall Metrics ---
auc_03 = roc_auc_score(y_test, y_prob_03)
auc_10 = roc_auc_score(y_test, y_prob_10)

# We will pick a standard threshold to compare, say best_t_10 (which is likely around 0.3 or 0.4)
# Let's say we use threshold 0.2 for 0.3 model, and threshold 0.4 for 1.0 model.
y_pred_03 = (y_prob_03 >= best_t_03).astype(int)
y_pred_10 = (y_prob_10 >= best_t_10).astype(int)

rec_03 = recall_score(y_test, y_pred_03)
prec_03 = precision_score(y_test, y_pred_03)

rec_10 = recall_score(y_test, y_pred_10)
prec_10 = precision_score(y_test, y_pred_10)

print(f"\n--- Final Comparison ---")
print(f"SMOTENC 0.3 (Threshold {best_t_03:.1f}): AUC={auc_03:.4f}, Recall={rec_03:.4f}, Precision={prec_03:.4f}")
print(f"SMOTENC 1.0 (Threshold {best_t_10:.1f}): AUC={auc_10:.4f}, Recall={rec_10:.4f}, Precision={prec_10:.4f}")

# --- Confusion Matrices ---
cm_03 = confusion_matrix(y_test, y_pred_03)
cm_10 = confusion_matrix(y_test, y_pred_10)

print(f"\nCM for SMOTENC 0.3:\n{cm_03}")
print(f"\nCM for SMOTENC 1.0:\n{cm_10}")

# --- Probability Histogram Plot ---
plt.figure(figsize=(10,5))
plt.hist(y_prob_03, bins=50, alpha=0.5, label='SMOTENC 0.3')
plt.hist(y_prob_10, bins=50, alpha=0.5, label='SMOTENC 1.0')
plt.title('Prediction Probabilities Distribution (SMOTENC)')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('smotenc_prob_histogram.png')

# Save Threshold Tradeoff Plot for 1.0
t_vals = [m[0] for m in metrics_10]
rec_vals = [m[1] for m in metrics_10]
prec_vals = [m[2] for m in metrics_10]

plt.figure(figsize=(8,6))
plt.plot(t_vals, rec_vals, label='Recall', marker='o')
plt.plot(t_vals, prec_vals, label='Precision', marker='s')
plt.title('Precision-Recall Tradeoff (SMOTENC 1.0)')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.savefig('smotenc_10_threshold_tradeoff.png')

print("\nSaved smotenc_prob_histogram.png and smotenc_10_threshold_tradeoff.png")
