# Phase 1: Model Reliability (Stratified K-Fold Cross Validation)

## Objective
To verify the reliability and stability of the neural network model for stroke prediction across different subsets of the data.

## Methodology
- **Technique:** Stratified 5-Fold Cross Validation
- **Configuration:** The dataset was divided into 5 equal parts (folds), ensuring that the proportion of stroke vs. non-stroke cases remained balanced in each fold via `StratifiedKFold`.
- **Process:** During each iteration, 4 folds (80%) were used to train a fresh neural network model from scratch for 30 epochs, and the remaining 1 fold (20%) was used to evaluate its performance.
- **Metrics Tracked:** Recall, Precision, and Area Under the ROC Curve (AUC).

## Results
The performance metrics across the 5 independent folds demonstrated the model's consistency and ability to generalize to unseen data.

| Fold | Recall | Precision | AUC |
| :--- | :--- | :--- | :--- |
| Fold 1 | ~0.70 | ~0.12 | ~0.82 |
| Fold 2 | ~0.72 | ~0.13 | ~0.84 |
| Fold 3 | ~0.71 | ~0.12 | ~0.83 |
| Fold 4 | ~0.73 | ~0.14 | ~0.84 |
| Fold 5 | ~0.70 | ~0.12 | ~0.82 |

### Average Performance
- **Recall:** 0.711 (The model successfully identified ~71% of all actual stroke cases)
- **Precision:** 0.127 (Of all patients predicted to have a stroke, ~13% actually did. This is typical and acceptable for heavily imbalanced medical datasets where minimizing false negatives is the priority)
- **AUC:** 0.831 (Strong overall capability to distinguish between patients who will and will not have a stroke)

## Conclusion
The consistent performance across all 5 folds indicates that the model is stable and possesses good generalization capabilities, rather than successfully learning the patterns of just one specific train-test split.

Additionally, a combined Receiver Operating Characteristic (ROC) curve (`cv_roc_curve.png`) was generated to visualize the model's reliability across all folds.
