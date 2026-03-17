import tensorflow as tf
import pickle
from sklearn.metrics import classification_report, confusion_matrix

# Load preprocessed data
with open('processed_data/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_test = data['X_test']
y_test = data['y_test']

# Load model
model = tf.keras.models.load_model('stroke_nn_model.h5')

# Evaluate
print("--- Evaluation Results ---")
eval_results = model.evaluate(X_test, y_test, verbose=0)
metrics_names = model.metrics_names
for name, val in zip(metrics_names, eval_results):
    print(f"{name.capitalize()}: {val:.4f}")

# Detailed Report
y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
