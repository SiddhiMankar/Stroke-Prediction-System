import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Phase 7 — User-Facing Stroke Prediction System
# ─────────────────────────────────────────────

# ─── Step 1: Load Saved Model and Scaler ───────────────────────────────────
print("Loading model and scaler...")
model = load_model('processed_data/stroke_final_model.h5')
scaler = joblib.load('processed_data/scaler.pkl')

# Load feature names to correctly order input
import pickle
with open('processed_data/preprocessed_data.pkl', 'rb') as f:
    meta = pickle.load(f)

feature_names = meta['feature_names']

# ─── Step 2: Risk Level Function ───────────────────────────────────────────
def get_risk_level(prob):
    if prob < 0.3:
        return "🟢 Low Risk"
    elif prob < 0.6:
        return "🟡 Moderate Risk"
    else:
        return "🔴 High Risk"

# ─── Step 3: Display Welcome Banner ────────────────────────────────────────
print("\n" + "="*50)
print("    STROKE RISK PREDICTION SYSTEM")
print("="*50)
print("Please answer the following questions about the patient.")
print("This tool is for educational/research purposes only.\n")

# ─── Step 4: Accept User Input ─────────────────────────────────────────────
try:
    age = float(input("Age: "))
    hypertension = int(input("Hypertension (0 = No, 1 = Yes): "))
    heart_disease = int(input("Heart Disease (0 = No, 1 = Yes): "))
    avg_glucose = float(input("Average Glucose Level (mg/dL): "))
    bmi = float(input("BMI: "))

    # For one-hot encoded features, collect them too
    print("\nGender (0 = Female, 1 = Male): ", end="")
    gender_male = int(input())

    print("Work Type:")
    print("  Options - 0: Never worked, 1: Private, 2: Self-employed, 3: Children")
    work_input = int(input("Enter selection (0-3): "))
    work_Never_worked = 1 if work_input == 0 else 0
    work_Private = 1 if work_input == 1 else 0
    work_Self_employed = 1 if work_input == 2 else 0
    work_children = 1 if work_input == 3 else 0

    print("Smoking Status (0: formerly smoked, 1: never smoked, 2: smokes, 3: unknown): ", end="")
    smoke_input = int(input())
    smoke_formerly = 1 if smoke_input == 0 else 0
    smoke_never = 1 if smoke_input == 1 else 0
    smoke_smokes = 1 if smoke_input == 2 else 0

    print("Residence Type (0 = Rural, 1 = Urban): ", end="")
    residence_urban = int(input())

    print("Ever Married? (0 = No, 1 = Yes): ", end="")
    ever_married_yes = int(input())

    # ─── Step 5: Build Patient Feature Dict Matching Training Order ────────
    patient_dict = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'avg_glucose_level': avg_glucose,
        'bmi': bmi,
        'gender_Male': gender_male,
        'work_type_Never_worked': work_Never_worked,
        'work_type_Private': work_Private,
        'work_type_Self-employed': work_Self_employed,
        'work_type_children': work_children,
        'smoking_status_formerly smoked': smoke_formerly,
        'smoking_status_never smoked': smoke_never,
        'smoking_status_smokes': smoke_smokes,
        'Residence_type_Urban': residence_urban,
        'ever_married_Yes': ever_married_yes
    }

    # Build array that matches training feature order
    patient_df = pd.DataFrame([patient_dict])
    # Reindex to match train column order exactly, fill any missing with 0
    patient_df = patient_df.reindex(columns=feature_names, fill_value=0)

    # ─── Step 6: Apply Preprocessing ───────────────────────────────────────
    patient_scaled = scaler.transform(patient_df)

    # ─── Step 7: Predict Stroke Probability ────────────────────────────────
    probability = model.predict(patient_scaled, verbose=0)[0][0]

    # ─── Step 8 & 9: Display Final Result ──────────────────────────────────
    print("\n" + "="*50)
    print("        PREDICTION RESULT")
    print("="*50)
    print(f"Stroke Probability : {probability:.1%}")
    print(f"Risk Category      : {get_risk_level(probability)}")
    print("="*50)
    print("\n⚠️  This is an AI-based research tool. Always consult a licensed medical professional.")

except ValueError as e:
    print(f"\n❌ Invalid input: {e}. Please enter numeric values where required.")
