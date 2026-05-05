import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('processed_data/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
feature_names = data['feature_names']
input_size = len(feature_names)

print("--- Class Distribution ---")
print(f"Before SMOTE: {pd.Series(y_train).value_counts().to_dict()}")

# Strategy 0.3
smote_03 = SMOTE(sampling_strategy=0.3, random_state=42)
X_res_03, y_res_03 = smote_03.fit_resample(X_train, y_train)
print(f"After SMOTE 0.3: {pd.Series(y_res_03).value_counts().to_dict()}")

# Build Model function
def build_model():
    model = Sequential([
        Input(shape=(input_size,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

# Function to test a strategy
def test_strategy(strat_val):
    smote = SMOTE(sampling_strategy=strat_val, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    classes = np.unique(y_res)
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_res)
    cw = dict(zip(classes, weights))
    
    model = build_model()
    history = model.fit(X_res, y_res, epochs=30, batch_size=32, class_weight=cw, verbose=0, validation_data=(X_test, y_test))
    
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred_1 = (y_prob >= 0.1).astype(int)
    y_pred_5 = (y_prob >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_prob)
    rec_1 = recall_score(y_test, y_pred_1)
    prec_1 = precision_score(y_test, y_pred_1)
    rec_5 = recall_score(y_test, y_pred_5)
    prec_5 = precision_score(y_test, y_pred_5)
    
    return history, y_prob, auc, rec_1, prec_1, rec_5, prec_5, confusion_matrix(y_test, y_pred_1)

# Run for 0.3, 0.5, 1.0
hist_03, prob_03, auc_03, r1_03, p1_03, r5_03, p5_03, cm_03 = test_strategy(0.3)
hist_05, prob_05, auc_05, r1_05, p1_05, r5_05, p5_05, cm_05 = test_strategy(0.5)
hist_10, prob_10, auc_10, r1_10, p1_10, r5_10, p5_10, cm_10 = test_strategy(1.0)

print("\n--- Strategy Comparison ---")
print(f"0.3 -> AUC: {auc_03:.4f}, Rec@0.1: {r1_03:.4f}, Prec@0.1: {p1_03:.4f}")
print(f"0.5 -> AUC: {auc_05:.4f}, Rec@0.1: {r1_05:.4f}, Prec@0.1: {p1_05:.4f}")
print(f"1.0 -> AUC: {auc_10:.4f}, Rec@0.1: {r1_10:.4f}, Prec@0.1: {p1_10:.4f}")

print("\n--- CM at 0.1 (Strategy 0.3) ---")
print(cm_03)

# Plots
# 1. Histogram
plt.figure(figsize=(8,5))
plt.hist(prob_03, bins=50, color='blue', alpha=0.7)
plt.title('Prediction Probabilities Distribution (SMOTE 0.3)')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.savefig('prob_histogram.png')

# 2. Training Curves for 0.3
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(hist_03.history['loss'], label='Train Loss')
plt.plot(hist_03.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1,2,2)
plt.plot(hist_03.history['auc'], label='Train AUC')
plt.plot(hist_03.history['val_auc'], label='Val AUC')
plt.title('AUC Curve')
plt.legend()
plt.savefig('training_curves_qna.png')

print("\nSaved plots to prob_histogram.png and training_curves_qna.png")
