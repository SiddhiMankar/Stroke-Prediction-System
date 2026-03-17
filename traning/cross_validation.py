import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, roc_auc_score, roc_curve

# 1. Load Preprocessed Data
data_path = 'processed_data/preprocessed_data.pkl'
print(f"Loading preprocessed data from {data_path}...")
with open(data_path, 'rb') as f:
    data = pickle.load(f)

# Combine the train/test splits back together since StratifiedKFold will handle the splitting
X = np.vstack((data['X_train'], data['X_test']))
y = np.concatenate((data['y_train'], data['y_test']))
class_weights = data['class_weights']
feature_names = data['feature_names']
input_size = len(feature_names)

print(f"Total dataset size: {X.shape[0]} samples")
print(f"Input size (number of features): {input_size}")

# Define function to build the MLP model
def build_mlp_model():
    model = Sequential([
        Input(shape=(input_size,)),
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
    return model

# Step 1 & 2 - Define K-Fold Configuration
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Step 3 - Prepare Metric Storage
recall_scores = []
precision_scores = []
auc_scores = []

# For plotting ROC curve later
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

print("\n--- Starting Stratified 5-Fold Cross Validation ---")

# Step 4 - Start the Cross Validation Loop
fold_no = 1
for train_index, val_index in skf.split(X, y):
    print(f"\nTraining Fold {fold_no}...")
    
    # Step 5 - Split Data for Each Fold
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Step 6 - Rebuild the Neural Network Each Time
    model = build_mlp_model()
    
    # Step 7 - Train the Model
    model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        class_weight=class_weights,
        verbose=0 # Make it silent as per requirements
    )
    
    # Step 8 - Generate Predictions
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Step 9 - Calculate Metrics
    recall = recall_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_prob)
    
    # For ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(auc)
    
    # Step 10 - Store the Metrics
    recall_scores.append(recall)
    precision_scores.append(precision)
    auc_scores.append(auc)
    
    # Step 11 - Print Fold Result
    print(f"Fold {fold_no} Result -> Recall: {recall:.3f}, Precision: {precision:.3f}, AUC: {auc:.3f}")
    
    fold_no += 1

# Step 12 - Compute Average Performance
print("\n--- Average Performance ---")
print(f"Recall:    {np.mean(recall_scores):.3f}")
print(f"Precision: {np.mean(precision_scores):.3f}")
print(f"AUC:       {np.mean(auc_scores):.3f}")

# Step 13 - Create Result Table
results = pd.DataFrame({
    "Fold": range(1, 6),
    "Recall": recall_scores,
    "Precision": precision_scores,
    "AUC": auc_scores
})
print("\nFinal Output Table:")
print(results)

# Saving the ROC Curve
plt.figure(figsize=(8, 6))

for i in range(5):
    plt.plot(mean_fpr, tprs[i], alpha=0.3, label=f'ROC fold {i+1} (AUC = {aucs[i]:.2f})')

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color='b',
         label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
         lw=2, alpha=.8)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - 5-Fold CV')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('cv_roc_curve.png')
print("\nSaved ROC curve to 'cv_roc_curve.png'")
