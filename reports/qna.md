# Q&A: SMOTE Implementation Details

**1. Exactly where in the pipeline did you apply SMOTE? (I want the exact order: split → scale → SMOTE → train)**
The exact order of operations was:
1. `train_test_split` (Isolating the test set completely)
2. `StandardScaler` (Scaling training data independently)
3. `SMOTE` (Applied exclusively to the scaled training data)
4. `model.fit` (Training the model on the SMOTE-augmented, scaled training data)

**2. Did you apply SMOTE before or after scaling?**
After scaling. SMOTE calculates Euclidean distances between minority points. Applying scaling first ensures that distance calculations are not skewed by features with larger magnitudes (like `avg_glucose_level` vs `age`).

**3. Confirm that SMOTE was NOT applied to test data — show code.**
Confirmed. The test data remains untouched. Here is the exact snippet from `data_preparation.py`:
```python
# 6. Apply Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Scaling applied to test data

# SMOTE applied ONLY to X_train_scaled
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Save dict ensuring test set uses non-SMOTE X_test_scaled
data_to_save = {
    'X_train_resampled': X_train_resampled,
    'X_test': X_test_scaled, 
    ...
```

**4. What was the class distribution before and after SMOTE?**
- **Before SMOTE:** `{0: 3901, 1: 187}` (Extreme ~20:1 imbalance)
- **After SMOTE (Strategy 0.3):** `{0: 3901, 1: 1170}` (Controlled ~3.3:1 imbalance)

**5. Did you use class weights during training after applying SMOTE?**
Yes.

**6. If yes, what were the class weight values?**
The newly computed class weights based on the `0.3` resampled distribution were:
- Class 0: `0.65`
- Class 1: `2.17`

**7. Why did you choose to combine SMOTE with class weights?**
Because `sampling_strategy=0.3` deliberately leaves a roughly 3:1 imbalance to prevent overfitting. We combined physical minority augmentation (SMOTE) with algorithmic penalty (class weights) so the model still heavily prioritizes the minority class (stroke cases) without having to memorize thousands of synthetic data points.

**8. What is the distribution of predicted probabilities after SMOTE?**
The probabilities are heavily skewed towards 0 (since the vast majority of cases are still negative), with a long, thin tail extending towards 1.

**9. Plot histogram of prediction probabilities.**
A histogram (`prob_histogram.png`) has been generated and saved to the project directory. It visually demonstrates the right-skewed distribution.

**10. Are predictions clustered around 0.5?**
No. They are clustered closer to 0.1 - 0.2. This is expected because the dataset remains imbalanced (3.3:1), so the model naturally outputs lower probabilities for the minority class, necessitating our threshold recalibration down to ~0.1 - 0.2.

**11. Was SMOTE applied independently inside each CV fold?**
Yes. Within `cross_validation.py`, `smote.fit_resample()` is called explicitly *inside* the `for train_index, val_index in skf.split(X, y):` loop, only on `X_train, y_train`.

**12. Was the model reinitialized for each fold?**
Yes. `model = build_mlp_model()` is called inside the CV loop to ensure a fresh, un-trained architecture for every fold.

**13. Were metrics averaged correctly across folds?**
Yes. Recall, Precision, and AUC were appended to arrays per-fold and then averaged using `np.mean()`.

**14. Did you check whether synthetic samples are realistic?**
SMOTE inherently creates synthetic points along the vector lines connecting existing minority points. Because it operates on standard-scaled data, the points exist safely within the bounds of the standardized feature space. However, see the next point regarding categorical variables.

**15. Were any categorical features interpolated incorrectly?**
Yes. Because we used standard `SMOTE` rather than `SMOTENC` (SMOTE for Nominal and Continuous), one-hot encoded features (like `gender_Male` or `ever_married_Yes`) were interpolated as continuous variables. 

**16. How were one-hot encoded features handled during SMOTE?**
They were treated as floats. Consequently, a synthetic sample might possess a fractional categorical value (e.g., `gender_Male = 0.6`). Neural networks handle fractional inputs gracefully, but theoretically, it implies "60% male", which is physically unrealistic.

**17. How are probabilities being converted to predictions?**
Via a boolean cast to integer: `y_pred = (y_probs >= thresh).astype(int)`.

**18. Is threshold applied correctly (>= vs >)?**
Yes, it is applied as `>=`.

**19. Show confusion matrix at threshold 0.1.**
Using the `0.3` SMOTE strategy at a `0.1` threshold:
```
[[804 156]
 [ 36  26]]
```
*(True Negatives: 804, False Positives: 156, False Negatives: 36, True Positives: 26)*

**20. Was early stopping used?**
No. The number of epochs was fixed at `30` as requested.

**21. Did loss converge properly?**
Yes. The loss dropped steeply within the first 10 epochs and plateaued nicely without heavy divergence between training and validation loss.

**22. Show training curves (loss + AUC).**
A plot (`training_curves_qna.png`) showing both Loss and AUC convergence has been generated and saved to the project root.

**23. Why was sampling_strategy=0.3 chosen?**
To mitigate overfitting. Creating roughly 3,700 synthetic copies from only 187 real stroke cases (a 1:1 strategy) risks creating highly repetitive, localized data clusters that the neural network simply memorizes. `0.3` strikes a balance between augmenting the minority class and retaining data variance.

**24. What happens with 0.5 and 1.0?**
Tests were run for 0.5 and 1.0 strategies:
- Increasing the SMOTE ratio slightly increases the overall AUC.
- However, paradoxically, it slightly lowers extreme threshold Recall (like at 0.1). Because higher SMOTE ratios push the *average* predicted probability higher, the relative distribution shifts, meaning threshold 0.1 behaves differently across different strategies.

**25. Compare results across these values.**
- **Strategy 0.3**: AUC: `0.7560`, Recall@0.1: `0.4194`, Precision@0.1: `0.1429`
- **Strategy 0.5**: AUC: `0.7651`, Recall@0.1: `0.3387`, Precision@0.1: `0.1419`
- **Strategy 1.0**: AUC: `0.7675`, Recall@0.1: `0.3226`, Precision@0.1: `0.1399`

**26. Were all other variables kept constant between pre-SMOTE and post-SMOTE runs?**
Yes. 

**27. Same architecture? Same epochs? Same random seed?**
- Architecture: `64 -> 32 -> 16` (Same)
- Epochs: `30` (Same)
- Random Seed: `42` for train-test split, StratifiedKFold, and SMOTE.

**28. Is there any step where training data information leaks into validation or test sets?**
No. Leakage is completely prevented because:
1. `train_test_split` occurs at the very beginning of the pipeline.
2. `StandardScaler` fits *only* on `X_train`.
3. `SMOTE` is applied *only* to `X_train`.
4. Inside cross-validation, SMOTE is strictly restricted to `X_train_fold` *after* the split has occurred.
