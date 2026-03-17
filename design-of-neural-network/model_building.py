import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

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

# 🔹 STEP 1 — Define Input Size
input_size = len(feature_names)
print(f"Input size (number of features): {input_size}")

# 🔹 STEP 2 — Build MLP Architecture
print("\n--- Building MLP Architecture ---")
model = Sequential([
    Input(shape=(input_size,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

# 🔹 STEP 3 — Compile the Model
print("\n--- Compiling Model ---")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# 🔹 STEP 4 — Train the Model
print("\n--- Training Model ---")
# Convert class weights to the format expected by Keras
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    verbose=1
)

# 🔹 STEP 4.1 — Print Final Training Metrics
final_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_auc = history.history['auc'][-1]
final_val_auc = history.history['val_auc'][-1]

print(f"\nFinal Training Metrics (Last Epoch):")
print(f"Loss: {final_loss:.4f} | Val Loss: {final_val_loss:.4f}")
print(f"AUC:  {final_auc:.4f} | Val AUC:  {final_val_auc:.4f}")

# 🔹 STEP 5 — Monitor During Training
# plotting history
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot AUC
plt.subplot(1, 2, 2)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Val AUC')
plt.title('AUC over Epochs')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
print("\nTraining curves saved as 'training_curves.png'")

# 🔹 STEP 6 — Evaluate on Test Set
print("\n--- Evaluating Model on Test Set ---")
eval_results = model.evaluate(X_test, y_test, verbose=0)
metrics_names = model.metrics_names
for name, val in zip(metrics_names, eval_results):
    print(f"{name.capitalize()}: {val:.4f}")

# Get predictions for confusion matrix
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Stroke', 'Stroke'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Confusion matrix plot saved as 'confusion_matrix.png'")

# Save the model
model.save('stroke_nn_model.h5')
print("\nModel saved as 'stroke_nn_model.h5'")
