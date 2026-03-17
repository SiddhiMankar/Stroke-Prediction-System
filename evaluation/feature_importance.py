import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import shap
import warnings
warnings.filterwarnings('ignore')

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

# Convert arrays to DataFrames for easier handling with feature names
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# Rebuild the selected Model (64 -> 32 -> 16)
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
# Using the best parameters from previous phases
model.fit(
    X_train, y_train,
    epochs=30, batch_size=32,
    class_weight=data['class_weights'],
    verbose=0
)

# --- SKLEarn Wrapper for Keras ---
# Permutation importance requires a model with a standard scikit-learn interface
from sklearn.base import BaseEstimator, ClassifierMixin

class KerasWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, keras_model):
        self.keras_model = keras_model
        self.classes_ = np.array([0, 1])
    
    def fit(self, X, y, **kwargs):
        return self
        
    def predict(self, X):
        return (self.keras_model.predict(X, verbose=0) > 0.4).astype(int).flatten()

wrapped_model = KerasWrapper(model)

# 3. Apply Permutation Importance
print("\n--- Running Permutation Importance ---")
result = permutation_importance(
    wrapped_model,
    X_test_df,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring='recall' # We prioritize recall in earlier phases
)

importance = result.importances_mean

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=True # Sort ascending for horizontal bar chart
)

print("\nFeature Importance Rankings:")
print(feature_importance.sort_values(by="Importance", ascending=False).to_string(index=False))

# 4. Visualize Permutation Importance
plt.figure(figsize=(10, 8))
plt.barh(feature_importance["Feature"], feature_importance["Importance"], color='skyblue')
plt.title("Permutation Feature Importance for Stroke Prediction")
plt.xlabel("Importance Score (Impact on Recall when feature is shuffled)")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig('permutation_importance.png')
print("Saved visualization to 'permutation_importance.png'")


# 7. Advanced Step: SHAP Values
print("\n--- Running SHAP Analysis ---")
# Use a background explainer for Deep Learning models
# Taking a random sample of 100 for background to speed up SHAP calculation
background = shap.sample(X_train_df.values, 100)

explainer = shap.Explainer(model.predict, background)
# Calculate SHAP values for the test set
# Using a subset of X_test if it's very large to save time
shap_values = explainer(X_test_df.values[:200])

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_df.head(200), show=False)
plt.savefig('shap_summary.png', bbox_inches='tight')
print("Saved SHAP visualization to 'shap_summary.png'")

print("\nPhase 4 completed successfully.")
