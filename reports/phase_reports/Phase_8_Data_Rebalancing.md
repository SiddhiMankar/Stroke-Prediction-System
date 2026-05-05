# Phase 8 — Data Rebalancing with SMOTE

## Objective
Improve the model’s ability to detect stroke cases by addressing severe class imbalance using synthetic data generation (SMOTE), while maintaining proper evaluation integrity.

## Input State
- Clean dataset already prepared.
- Train-test split already exists.
- Class weights already implemented.
- Model previously trained on imbalanced data.

## Actions Taken
1. **Installed Dependency:** Installed `imbalanced-learn`.
2. **Modified `data_preparation.py`:**
   - Imported `SMOTE` from `imblearn.over_sampling`.
   - Applied **Controlled SMOTE** (`sampling_strategy=0.3, random_state=42`) exclusively to the training data (`X_train_scaled`, `y_train`) after the train-test split to prevent data leakage.
   - Saved both the original and resampled training data (`X_train_resampled`, `y_train_resampled`) to `preprocessed_data.pkl` to allow cross-validation to work correctly without data leakage.
3. **Preserved Test Data Integrity:** Ensure `X_test` and `y_test` remained completely untouched and untransformed by SMOTE.

## Interpretation Logic
"SMOTE improves minority class representation, enabling the model to better detect stroke cases without requiring additional real-world data. By using Controlled SMOTE (0.3 strategy), we avoid full 1:1 balancing, maintaining a realistic class distribution while reducing the risk of overfitting."
