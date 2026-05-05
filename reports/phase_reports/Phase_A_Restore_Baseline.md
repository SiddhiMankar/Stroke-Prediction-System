# Phase A — Restore Baseline Model (Final Production Model)

## Objective
Revert to the original, stable class-weighted learning configuration after validating that synthetic oversampling (SMOTE/SMOTENC) failed to yield robust improvements.

## Input State
- Experimental SMOTE and SMOTENC pipelines active but underperforming.

## Actions Taken
1. **Removed Resampling Logic**:
   - Stripped all `imblearn.over_sampling` logic (both SMOTE and SMOTENC) from `data_preparation.py`, `cross_validation.py`, and `train_final_model.py`.
2. **Restored Class Weights**:
   - Re-introduced `compute_class_weight(class_weight='balanced', ...)` based purely on the true, imbalanced `y_train` dataset.
3. **Model Architecture Maintained**:
   - `64 -> 32 -> 16` dense architecture.
   - Optimizer: `adam`
   - Loss: `binary_crossentropy`
   - Epochs: `30`
4. **Retrained Final Model**:
   - The final model was re-trained successfully purely on the real dataset using the restored class weights.

## Interpretation Logic
"Class-weighted learning provides better generalization than synthetic oversampling for this dataset. By forcing the neural network to heavily penalize errors on actual minority class points rather than synthesizing artificial ones, we avoid the catastrophic overfitting seen during the SMOTE/SMOTENC trials."
