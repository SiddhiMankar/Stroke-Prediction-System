# Phase 10R — Retrain Final Model (Clean Version)

## Objective
Train two final models utilizing the cleaned `SMOTENC` strategy at two different resampling ratios (0.3 and 1.0) for a thorough comparison.

## Input State
- Unscaled `X_train` loaded.

## Actions Taken
1. **Applied Preprocessing**:
   - Scaled `X_train` with `StandardScaler`.
2. **Resampled**:
   - Generated `X_res_03` using `SMOTENC(sampling_strategy=0.3)`.
   - Generated `X_res_10` using `SMOTENC(sampling_strategy=1.0)`.
3. **Trained Models**:
   - Model 1 (0.3) was trained and saved to `stroke_final_model_03.h5`.
   - Model 2 (1.0) was trained and saved to `stroke_final_model_10.h5`.
   - Both models were trained without class weights.

## Interpretation Logic
"The corrected resampling strategy allows the neural network to learn meaningful decision boundaries without distortion. Building two separate models helps us scientifically validate which sampling ratio best serves this specific neural architecture."
