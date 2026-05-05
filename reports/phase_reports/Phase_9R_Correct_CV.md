# Phase 9R — Correct Cross-Validation with SMOTENC

## Objective
Update the cross-validation logic to scale features properly inside each fold, followed by the application of SMOTENC.

## Input State
- CV previously used standard SMOTE.

## Actions Taken
1. **Inside CV Loop**:
   - For each fold split (`train_index`, `val_index`), the `StandardScaler` was applied exclusively to `X_train_fold`.
   - `X_val_fold` was subsequently transformed using this fitted scaler to prevent data leakage.
   - `SMOTENC` was then applied exclusively to the scaled `X_train_fold`.
2. **Removed Class Weights**:
   - `class_weight` was removed from the `model.fit()` call inside the CV loop.

## Interpretation Logic
"Balanced sampling within each fold ensures unbiased validation and consistent minority class learning. Scaling before SMOTENC correctly structures the numerical distances without corrupting the explicit discrete tracking of categorical features handled by SMOTENC."
