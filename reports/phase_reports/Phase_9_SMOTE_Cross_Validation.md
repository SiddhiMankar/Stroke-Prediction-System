# Phase 9 — SMOTE in Cross-Validation

## Objective
Validate the model's performance reliably across multiple folds using SMOTE within the cross-validation loop to avoid bias and data leakage.

## Input State
- Stratified K-Fold cross-validation already implemented.
- Currently operating on imbalanced folds (prior to this phase).

## Actions Taken
1. **Modified `cross_validation.py`:**
   - Imported `SMOTE`.
   - Updated the cross-validation loop to apply SMOTE (`random_state=42`) *only* to the training subset (`X_train`, `y_train`) within each fold, immediately after the fold split.
   - Ensured the validation fold (`X_val`, `y_val`) was never exposed to synthetic data.
2. **Results Obtained:**
   - Average Recall: 0.482
   - Average Precision: 0.115
   - Average AUC: 0.758

## Interpretation Logic
"Applying SMOTE within each fold ensures balanced learning across all validation splits, preventing bias toward the majority class and ensuring that our validation metrics reflect true generalization performance."
