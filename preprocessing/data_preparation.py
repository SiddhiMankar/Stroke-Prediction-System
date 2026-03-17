import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import os
import pickle
import joblib

# Load dataset
data_path = os.path.join('..', 'data', 'stroke_cleaned_final.csv')
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)

# 1. Handle Missing BMI
if df['bmi'].isnull().any():
    median_bmi = df['bmi'].median()
    df['bmi'].fillna(median_bmi, inplace=True)

# 2. Remove ID if exists
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# 3. Handle specific categorical encoding to ensure 16 features
# --- One-hot encoding ---
categorical_cols = ['gender', 'work_type', 'smoking_status', 'Residence_type', 'ever_married']
existing_cat_cols = [col for col in categorical_cols if col in df.columns]

# One-hot encode
df_encoded = pd.get_dummies(df.drop(columns=['age_group', 'bmi_category']), columns=existing_cat_cols, drop_first=True)

# 4. Separate Features and Target
print("\n--- Feature Preparation ---")
X = df_encoded.drop('stroke', axis=1)
y = df_encoded['stroke']

# Convert to float
X = X.astype(float)
feature_names = X.columns.tolist()
print(f"Number of features: {len(feature_names)}")
print(f"Features: {feature_names}")

print("\n--- Final dataset creation (First 5 rows of X) ---")
print(X.head())

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Apply Feature Scaling
# --- Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Compute Class Weights
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, weights))
print(f"Computed class weights: {class_weights_dict}")

# 8. Save data for Model Building phase
output_dir = os.path.join('preprocessing', 'processed_data')
os.makedirs(output_dir, exist_ok=True)

data_to_save = {
    'X_train': X_train_scaled,
    'X_test': X_test_scaled,
    'y_train': y_train,
    'y_test': y_test,
    'class_weights': class_weights_dict,
    'feature_names': feature_names
}

with open(os.path.join(output_dir, 'preprocessed_data.pkl'), 'wb') as f:
    pickle.dump(data_to_save, f)

# Also save the fitted scaler separately
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
print(f"✅ Scaler saved to '{os.path.join(output_dir, 'scaler.pkl')}'")

print(f"\n✅ Data preparation complete and saved to '{os.path.join(output_dir, 'preprocessed_data.pkl')}'")
