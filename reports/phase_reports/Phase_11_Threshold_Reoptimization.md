# Phase 11 — Threshold Re-optimization

## Objective
Recompute predictions across probability thresholds to identify the new optimal decision boundary after SMOTE, as the prediction probabilities inherently shift post-rebalancing.

## Input State
- Model retrained using SMOTE.
- Previous optimal threshold = 0.3 (now outdated).

## Actions Taken
1. **Updated `threshold_optimization.py`:**
   - Modified script to load the finalized `stroke_final_model.h5` rather than retraining from scratch.
   - Evaluated the model across the threshold range `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`.
2. **Results Obtained:**
   - At threshold `0.1`, Recall is `0.4677`, Precision is `0.1526`.
   - At threshold `0.2`, Recall is `0.3548`, Precision is `0.1476`.
   - The optimal threshold naturally shifted lower compared to the original, highly-imbalanced dataset.

## Interpretation Logic
"Post-rebalancing, prediction probabilities shift, requiring recalibration of decision thresholds to maintain optimal sensitivity. The optimal threshold shifts lower (around 0.1-0.2) to capture maximum true positives (stroke cases)."
