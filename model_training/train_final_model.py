import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# 1. Load Preprocessed Data (scaled)
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'data_processing', 'preprocessed_data.pkl')
print(f"Loading preprocessed data from {data_path}...")
with open(data_path, 'rb') as f:
    data = pickle.load(f)

X_train_scaled = data['X_train']
y_train = data['y_train']
X_test_scaled = data['X_test']
y_test = data['y_test']
class_weights = data['class_weights']
feature_names = data['feature_names']
input_size = len(feature_names)

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

# 2. Train Model
print("\n--- Training Final Baseline Model ---")
model = build_model()
model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, class_weight=class_weights, verbose=0)
path = os.path.join(BASE_DIR, 'models', 'final', 'stroke_final_model.h5')
model.save(path)

print(f"\nFinal model saved to '{path}'")
