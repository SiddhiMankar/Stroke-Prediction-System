# Phase 6 & 7: Risk Level Categorization and Prediction System

---

## Phase 6 — Risk Level Categorization

### Objective
To convert the neural network's raw probability score (a decimal like `0.63`) into a clinically interpretable risk label, making the model's output human-readable for doctors, patients, and examiners.

### Methodology
A simple mapping function was defined to classify stroke probabilities into three meaningful categories:

```python
def get_risk_level(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.6:
        return "Moderate Risk"
    else:
        return "High Risk"
```

| Probability Range | Risk Level |
| :--- | :--- |
| < 0.30 | 🟢 Low Risk |
| 0.30 – 0.60 | 🟡 Moderate Risk |
| > 0.60 | 🔴 High Risk |

### Why This Is Important
A neural network outputting `0.63` is meaningless to a non-technical user. Translating it to **"High Risk"** allows the system to function like a real clinical screening tool. This is a key requirement for any medical AI system — **interpretability**.

### Results
When applied to the full test set, the risk level distribution was observed. The model appropriately elevates risk labels for patients with high probability scores, and actual stroke patients were predominantly categorized as Moderate-to-High Risk, validating the thresholding approach.

---

## Phase 7 — Prediction System

### Objective
To convert the trained model from a pure research artifact into a practical, user-facing screening tool that can take a real patient's data as input and produce an instant risk assessment.

### Architecture
The prediction system (`predict_stroke.py`) follows this pipeline:

1. **Load saved artifacts** — Loads the final trained Keras model (`stroke_final_model.h5`) and the fitted data scaler (`scaler.pkl`) that was saved during data preparation.
2. **Collect patient input** — Prompts the user to enter all required clinical features: age, BMI, average glucose level, hypertension, heart disease, gender, work type, smoking status, residence type, and marital status.
3. **Preprocess inputs** — Applies the same `StandardScaler` transformation as was used on training data, ensuring the model receives identically formatted input.
4. **Predict probability** — Runs the model's forward pass and extracts the probability score.
5. **Display risk level** — Applies the `get_risk_level()` function from Phase 6 and presents a clean, color-coded result.

### Example Output
```
==================================================
        PREDICTION RESULT
==================================================
Stroke Probability  :  67.3%
Risk Category       :  🔴 High Risk
==================================================

⚠️  This is an AI-based research tool. Always consult a licensed medical professional.
```

### How to Run
```bash
python predict_stroke.py
```
The system will interactively prompt for all patient details and display the prediction instantly.

### Files Generated
| File | Purpose |
| :--- | :--- |
| `risk_categorization.py` | Phase 6 script — applies risk labels to test set and saves the final trained model |
| `predict_stroke.py` | Phase 7 script — the interactive, user-facing stroke risk screening tool |
| `processed_data/stroke_final_model.h5` | The saved final trained Keras model (64 → 32 → 16 architecture) |
| `processed_data/scaler.pkl` | The saved fitted StandardScaler for preprocessing new patient inputs |

### Conclusion
The system now functions as a complete end-to-end stroke screening pipeline — from raw patient data all the way to an understandable, medically framed risk assessment. This demonstrates practical, real-world applicability beyond just academic model training.
