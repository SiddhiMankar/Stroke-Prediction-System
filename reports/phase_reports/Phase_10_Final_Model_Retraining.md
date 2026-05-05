# Phase 10 — Final Model Retraining

## Objective
Train the final neural network architecture using the newly SMOTE-augmented data to maximize sensitivity in stroke risk detection.

## Input State
- SMOTE validated through cross-validation.
- Best architecture selected (64 → 32 → 16).

## Actions Taken
1. **Created `train_final_model.py`:**
   - Loaded the resampled training data (`X_train_resampled`, `y_train_resampled`) from `preprocessed_data.pkl`.
   - Rebuilt the selected neural network architecture (64 → 32 → 16).
   - Compiled the model with `adam` optimizer and `binary_crossentropy` loss.
   - Trained the model using the SMOTE-augmented dataset and class weights.
2. **Saved Final Model:**
   - Saved the resulting trained model as `stroke_final_model.h5` in the `processed_data/` directory for downstream thresholding and evaluation.

## Interpretation Logic
"Final model incorporates synthetic augmentation to maximize sensitivity in stroke risk detection, preparing it for real-world deployment."
