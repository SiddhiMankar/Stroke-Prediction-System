import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, roc_auc_score

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

print(f"Input size (number of features): {input_size}")

# 2. Define Model Architectures to Test
architectures = {
    "Model 1 (32 -> 16)": [32, 16],
    "Model 2 (64 -> 32)": [64, 32],
    "Model 3 (64 -> 32 -> 16)": [64, 32, 16],
    "Model 4 (128 -> 64)": [128, 64]
}

def build_model(layers):
    model = Sequential()
    model.add(Input(shape=(input_size,)))
    for units in layers:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
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

# 3. Storage for Results
results = []

print("\n--- Starting Phase 2: Hyperparameter Tuning ---")

# 4. Train and Evaluate Each Architecture
for model_name, layers in architectures.items():
    print(f"\nTraining {model_name}...")
    
    model = build_model(layers)
    
    # Train
    model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        verbose=0 # Make it silent
    )
    
    # Evaluate on validation data
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    print(f"Result -> Recall: {recall:.3f}, Precision: {precision:.3f}, AUC: {auc:.3f}")
    
    results.append({
        "Model": model_name,
        "Recall": recall,
        "Precision": precision,
        "AUC": auc
    })

# 5. Review Results
results_df = pd.DataFrame(results)

print("\n--- Hyperparameter Tuning Complete ---")
print("\nFinal Comparison Table:")
print(results_df.to_string(index=False, float_format="%.3f"))

results_df.to_csv("hyperparameter_tuning_results.csv", index=False)
print("\nSaved results to 'hyperparameter_tuning_results.csv'")
