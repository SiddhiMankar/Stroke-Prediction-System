import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

# 1. Load Preprocessed Data
data_path = 'processed_data/preprocessed_data.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# The data in the pkl seems to be already scaled based on my earlier check.
# Let's verify by checking the mean/std
print(f"X_train mean: {np.mean(X_train):.3f}, std: {np.std(X_train):.3f}")

# 2. Define Model Architectures
architectures = {
    "Model 1 (32, 16)": (32, 16),
    "Model 2 (64, 32)": (64, 32),
    "Model 3 (64, 32, 16)": (64, 32, 16),
    "Model 4 (128, 64)": (128, 64)
}

plt.figure(figsize=(10, 8))
colors = ['#FF4B2B', '#23D5AB', '#23A6D5', '#EE7752'] 

print("\n--- Training Surrogate Models (MLP) ---")

for (model_name, layers), color in zip(architectures.items(), colors):
    print(f"Training {model_name}...")
    
    # We use Adam solver and a small learning rate. 
    # To handle imbalance without imblearn, we'll use a larger alpha (regularization) 
    # and increase max_iter.
    model = MLPClassifier(hidden_layer_sizes=layers, 
                          activation='relu', 
                          solver='adam', 
                          alpha=1.0, # Strong regularization to handle noisy minority
                          learning_rate_init=0.001,
                          max_iter=500, 
                          random_state=42)
    
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    print(f"Result -> AUC: {roc_auc:.4f}")
    
    plt.plot(fpr, tpr, color=color, lw=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})')

# Add Random Guessing line
plt.plot([0, 1], [0, 1], color='#2c3e50', lw=1.5, linestyle='--', label='Random Guessing (AUC = 0.50)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Comparative ROC Curves: Neural Network Architectures', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig('comparative_roc_auc.png', dpi=300)
print("\nPlot saved as 'comparative_roc_auc.png'")
