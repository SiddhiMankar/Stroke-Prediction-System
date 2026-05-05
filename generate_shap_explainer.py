import os
import joblib
import pickle
import numpy as np
import shap
import tensorflow as tf

# -----------------------------------------------------------------------------
# Paths configuration
# -----------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(__file__, '..'))
# Inference expects a 'models/final' folder two levels up from this script
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final', 'stroke_final_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'final', 'scaler.pkl')
PREPROC_PATH = os.path.join(BASE_DIR, 'data_processing', 'preprocessed_data.pkl')
EXPLAINER_OUT = os.path.join(BASE_DIR, 'models', 'final', 'shap_explainer.pkl')

# -----------------------------------------------------------------------------
# Load artifacts
# -----------------------------------------------------------------------------
print(f"Loading model from {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)
print(f"Loading preprocessed data from {PREPROC_PATH}")
with open(PREPROC_PATH, 'rb') as f:
    data = pickle.load(f)
    X_train = data['X_train']  # already scaled? In our pipeline X_train is the *scaled* array
    # If X_train is not scaled, we would scale it here, but the training script already
    # saved the scaled version, so we can use it directly.

# -----------------------------------------------------------------------------
# Prepare background dataset for SHAP (a subset of training data)
# -----------------------------------------------------------------------------
# Use a random sample of up to 200 rows to keep explainer creation fast.
background = shap.sample(X_train, min(200, X_train.shape[0]))

# -----------------------------------------------------------------------------
# Build SHAP explainer
# -----------------------------------------------------------------------------
# For a Keras binary classifier we can use KernelExplainer (model.predict returns probability)
explainer = shap.KernelExplainer(model.predict, background)

# -----------------------------------------------------------------------------
# Save explainer
# -----------------------------------------------------------------------------
joblib.dump(explainer, EXPLAINER_OUT)
print(f"SHAP explainer saved to {EXPLAINER_OUT}")
