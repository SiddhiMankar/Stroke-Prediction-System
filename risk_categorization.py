import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pickle
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Phase 6 — Risk Level Categorization
# ─────────────────────────────────────────────

# Step 1: Define Risk Thresholds
def get_risk_level(prob):
    """Convert a raw probability score into a human-readable risk label."""
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.6:
        return "Moderate Risk"
    else:
        return "High Risk"

# ─────────────────────────────────────────────
# Load data and train model to demonstrate categorization
# ─────────────────────────────────────────────
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

# Rebuild the selected model (64 -> 32 -> 16)
print("\nRebuilding selected model (64 -> 32 -> 16)...")
model = Sequential([
    Input(shape=(input_size,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=32,
          class_weight=data['class_weights'], verbose=0)

# Save the final trained model for use in Phase 7
model.save('processed_data/stroke_final_model.h5')
print("✅ Final model saved to 'processed_data/stroke_final_model.h5'")

# Step 2: Apply risk categorization to all test predictions
print("\n--- Phase 6: Risk Level Categorization on Test Set ---")
y_probs = model.predict(X_test, verbose=0).flatten()

# Build a summary DataFrame
df_results = pd.DataFrame({
    'Stroke Probability': y_probs,
    'Risk Level': [get_risk_level(p) for p in y_probs],
    'Actual Stroke': y_test.values
})

# Step 3: Print example outputs
print("\nSample individual predictions:")
sample = df_results.sample(10, random_state=1)
for _, row in sample.iterrows():
    print(f"  Probability: {row['Stroke Probability']:.3f}  →  Risk Level: {row['Risk Level']}  (Actual: {'Stroke' if row['Actual Stroke'] == 1 else 'No Stroke'})")

# Step 4: Summarize distribution of risk levels
print("\nRisk Level Distribution in Test Set:")
print(df_results['Risk Level'].value_counts())

# Step 5: Check detection among actual stroke cases
print("\nRisk Level Distribution for ACTUAL Stroke Patients:")
print(df_results[df_results['Actual Stroke'] == 1]['Risk Level'].value_counts())
