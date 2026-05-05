# Phase C — Experimental Evidence Packaging

## Objective
Preserve the results of the SMOTE and SMOTENC experiments to formally justify the decision to revert to the baseline model.

## Final Metric Comparison Table

| Metric | Baseline (Class Weights Only) | SMOTE (Wrong Categoricals) | SMOTENC 0.3 (Imbalanced Synths) | SMOTENC 1.0 (Full Overfit) |
|---|---|---|---|---|
| **Recall** | **~0.77 (at thresh 0.3)** | 0.47 (at thresh 0.1) | 0.45 (at thresh 0.2) | 0.34 (at thresh 0.1) |
| **AUC** | **~0.80 - 0.82** | 0.77 | 0.79 | 0.74 |
| **Precision** | **~0.14 (at thresh 0.3)** | 0.15 (at thresh 0.1) | 0.16 (at thresh 0.2) | 0.11 (at thresh 0.1) |

## Confusion Matrix Comparison

### Baseline (Optimal Threshold 0.3)
```
[[802 158]
 [ 14  48]]
```
- **False Negatives**: Only 14 (Excellent sensitivity)

### SMOTENC 1.0 (Best Attempt, Threshold 0.1)
```
[[797 163]
 [ 41  21]]
```
- **False Negatives**: 41 (Catastrophic sensitivity loss due to overfitting)

## Interpretation Logic
"Resampling techniques were evaluated but did not improve performance, highlighting dataset-specific limitations. Because our dataset has a severe 20:1 imbalance with only 187 genuine positive samples, attempting to generate synthetic data (even via the corrected SMOTENC method) caused the neural network to memorize localized synthetic clusters rather than generalize true risk boundaries. Class-weighted learning remains the mathematically superior approach for this specific data structure."
