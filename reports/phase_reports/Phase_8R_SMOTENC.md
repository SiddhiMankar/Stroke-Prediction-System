# Phase 8R — Replace SMOTE with SMOTENC (Clean Recovery)

## Objective
Correct the data resampling phase by replacing standard SMOTE with `SMOTENC` to properly handle categorical features without generating unrealistic synthetic samples.

## Input State
- SMOTE previously applied (Incorrect for categorical variables).
- One-hot encoded features exist.
- Categorical features indices mapped correctly.

## Actions Taken
1. **Identified Categorical Features**: 
   - `age` (0), `avg_glucose_level` (3), `bmi` (4) are numerical.
   - All other 13 features (indices `1, 2, 5-15`) are categorical.
   - Defined `categorical_features = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`.
2. **Applied SMOTENC**:
   - We utilized `SMOTENC(categorical_features=categorical_features, sampling_strategy=0.3, random_state=42)` and `SMOTENC(..., random_state=42)` (ratio 1.0) to test multiple strategies.
3. **Removed Class Weights**:
   - The model was updated to train purely on the augmented dataset without auxiliary loss penalties to evaluate the raw impact of clean synthetic sampling.

## Interpretation Logic
"Using SMOTENC preserves categorical integrity, preventing unrealistic synthetic samples and improving model learning stability. By evaluating both 0.3 and 1.0 strategies, we test whether a balanced or slightly imbalanced approach provides better generalization."
