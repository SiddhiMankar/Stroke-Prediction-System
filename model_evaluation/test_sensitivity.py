import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

model = load_model('processed_data/stroke_final_model.h5')
scaler = joblib.load('processed_data/scaler.pkl')
with open('processed_data/preprocessed_data.pkl', 'rb') as f:
    meta = pickle.load(f)
feature_names = meta['feature_names']

def predict(p):
    df = pd.DataFrame([p])
    df = df.reindex(columns=feature_names, fill_value=0)
    print(f"DEBUG: Feature order: {df.columns.tolist()}")
    print(f"DEBUG: Values: {df.values[0]}")
    scaled = scaler.transform(df)
    print(f"DEBUG: Scaled: {scaled[0]}")
    prob = model.predict(scaled, verbose=0)[0][0]
    return prob

p_70 = {
    'age': 70, 'hypertension': 0, 'heart_disease': 1, 'avg_glucose_level': 300, 'bmi': 30,
    'gender_Male': 1, 'work_type_Self-employed': 1, 'smoking_status_smokes': 1, 'Residence_type_Urban': 0, 'ever_married_Yes': 1
}

p_50 = {
    'age': 50, 'hypertension': 1, 'heart_disease': 1, 'avg_glucose_level': 300, 'bmi': 35,
    'gender_Male': 1, 'work_type_Private': 1, 'smoking_status_smokes': 1, 'Residence_type_Urban': 0, 'ever_married_Yes': 1
}

print(f"Prob (70yo): {predict(p_70):.1%}")
print(f"Prob (50yo): {predict(p_50):.1%}")

# Test sensitivity
p_70_private = p_70.copy()
p_70_private['work_type_Self-employed'] = 0
p_70_private['work_type_Private'] = 1
print(f"Prob (70yo + Private Work): {predict(p_70_private):.1%}")

p_70_hyp_private = p_70_private.copy()
p_70_hyp_private['hypertension'] = 1
print(f"Prob (70yo + Hypertension + Private Work): {predict(p_70_hyp_private):.1%}")
