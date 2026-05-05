from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle
import joblib
import shap
import os

app = FastAPI(title="Stroke Prediction API", description="API for predicting stroke risk using the baseline class-weighted model")

# Configure CORS so the React frontend can talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory (one level up from 'inference')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the artifacts at startup
MODEL_PATH = os.path.join(BASE_DIR, "models", "final", "stroke_final_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "final", "scaler.pkl")
PREPROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data_processing", "preprocessed_data.pkl")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    with open(PREPROCESSED_DATA_PATH, 'rb') as f:
        data = pickle.load(f)
        feature_names = data['feature_names']
        # Load SHAP explainer (generated during training)
        SHAP_EXPLAINER_PATH = os.path.join(BASE_DIR, "models", "final", "shap_explainer.pkl")
        try:
            with open(SHAP_EXPLAINER_PATH, "rb") as f:
                shap_explainer = joblib.load(f)
        except Exception as e:
            print(f"Error loading SHAP explainer: {e}")
            shap_explainer = None
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    model = None
    scaler = None
    feature_names = None

class PatientData(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    gender: str # "Male", "Female", "Other"
    work_type: str # "Never_worked", "Private", "Self-employed", "children", "Govt_job"
    smoking_status: str # "formerly smoked", "never smoked", "smokes", "Unknown"
    residence_type: str # "Urban", "Rural"
    ever_married: str # "Yes", "No"
def _get_shap_contributions(input_scaled):
    """
    Compute SHAP values for a single input and return the most
    influential features with their direction (+/-) and magnitude.
    Returns a list of dicts:
    [
        {"feature": "age", "contribution": 0.12, "direction": "increase"},
        ...
    ]
    """
    if shap_explainer is None:
        return []
    # Obtain SHAP values (list for binary classifiers)
    shap_vals = np.array(shap_explainer.shap_values(input_scaled)).flatten()
    # Pair each SHAP value with its feature name
    contributions = []
    for i, val in enumerate(shap_vals):
        contributions.append({
            "feature": feature_names[i],
            "contribution": float(val),
            "direction": "increase" if val > 0 else "decrease"
        })
    # Sort by absolute impact and keep top 5
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions[:5]
@app.post("/predict")
def predict_stroke(patient: PatientData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded properly.")
    
    # Preprocess inputs into the exact 16 feature order:
    # ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 
    #  'gender_Male', 'gender_Other', 'work_type_Never_worked', 'work_type_Private', 
    #  'work_type_Self-employed', 'work_type_children', 'smoking_status_formerly smoked', 
    #  'smoking_status_never smoked', 'smoking_status_smokes', 'Residence_type_Urban', 'ever_married_Yes']
    
    # Start with numerical/direct features
    features = {
        'age': patient.age,
        'hypertension': patient.hypertension,
        'heart_disease': patient.heart_disease,
        'avg_glucose_level': patient.avg_glucose_level,
        'bmi': patient.bmi,
        
        # Initialize binary columns to 0
        'gender_Male': 0,
        'gender_Other': 0,
        'work_type_Never_worked': 0,
        'work_type_Private': 0,
        'work_type_Self-employed': 0,
        'work_type_children': 0,
        'smoking_status_formerly smoked': 0,
        'smoking_status_never smoked': 0,
        'smoking_status_smokes': 0,
        'Residence_type_Urban': 0,
        'ever_married_Yes': 0
    }
    
    # Gender
    if patient.gender == "Male":
        features['gender_Male'] = 1
    elif patient.gender == "Other":
        features['gender_Other'] = 1
        
    # Work Type
    if patient.work_type == "Never_worked":
        features['work_type_Never_worked'] = 1
    elif patient.work_type == "Private":
        features['work_type_Private'] = 1
    elif patient.work_type == "Self-employed":
        features['work_type_Self-employed'] = 1
    elif patient.work_type == "children":
        features['work_type_children'] = 1
        
    # Smoking Status
    if patient.smoking_status == "formerly smoked":
        features['smoking_status_formerly smoked'] = 1
    elif patient.smoking_status == "never smoked":
        features['smoking_status_never smoked'] = 1
    elif patient.smoking_status == "smokes":
        features['smoking_status_smokes'] = 1
        
    # Residence
    if patient.residence_type == "Urban":
        features['Residence_type_Urban'] = 1
        
    # Married
    if patient.ever_married == "Yes":
        features['ever_married_Yes'] = 1
        
    # Construct array in exact order
    input_array = np.array([[
        features['age'],
        features['hypertension'],
        features['heart_disease'],
        features['avg_glucose_level'],
        features['bmi'],
        features['gender_Male'],
        features['gender_Other'],
        features['work_type_Never_worked'],
        features['work_type_Private'],
        features['work_type_Self-employed'],
        features['work_type_children'],
        features['smoking_status_formerly smoked'],
        features['smoking_status_never smoked'],
        features['smoking_status_smokes'],
        features['Residence_type_Urban'],
        features['ever_married_Yes']
    ]])
    
    # Scale
    input_scaled = scaler.transform(input_array)
    
    # Predict
    prob = model.predict(input_scaled)[0][0]
    
    # Threshold is 0.3 as determined by threshold optimization
    threshold = 0.3
    risk_level = "High" if prob >= threshold else "Low"
    
    shap_contributions = _get_shap_contributions(input_scaled)
    return {
        "probability": float(prob),
        "risk_level": risk_level,
        "threshold": threshold,
        "shap_contributions": shap_contributions
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
