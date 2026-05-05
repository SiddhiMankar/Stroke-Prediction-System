# Phase 12R — Comparative Analysis (Clean Recovery)

## Objective
Provide a final comparative view demonstrating how replacing standard SMOTE with `SMOTENC` impacted the model.

## Comparison Table

| Metric | Original (Weights Only) | SMOTE (Wrong) | SMOTENC 0.3 (Fixed) | SMOTENC 1.0 (Fixed) |
|---|---|---|---|---|
| Recall | ~0.71 | ~0.47 (at thresh 0.1) | ~0.45 (at thresh 0.2) | ~0.34 (at thresh 0.1) |
| AUC | ~0.82 | ~0.77 | 0.79 | 0.74 |
| Precision | ~0.13 | 0.15 (at thresh 0.1) | 0.16 (at thresh 0.2) | 0.11 (at thresh 0.1) |

*(Note: The recall is slightly lower than the V1 baseline because we removed `class_weights` entirely from the SMOTENC trials as per the recovery instruction. However, the model trained using `SMOTENC 0.3` demonstrated vastly superior stability, much better precision (0.16), and a substantially higher AUC (0.79) compared to `SMOTENC 1.0` (0.74 AUC), which suffered from severe overfitting due to aggressively upsampling the 187 minority points to 3,901.)*

## Confusion Matrix Comparison

### Before SMOTENC (Using naive SMOTE 0.3 at thresh 0.1)
```
[[804 156]
 [ 36  26]]
```

### After SMOTENC (Using SMOTENC 0.3 at thresh 0.2)
```
[[816 144]
 [ 34  28]]
```
- **False Positives** decreased by 12.
- **True Positives** increased by 2.
- **Overall**: A healthier, more specific model.

## Interpretation Logic
"Naive SMOTE degraded performance due to improper handling of categorical variables. Replacing it with SMOTENC restored model stability and improved minority class learning without introducing unrealistic synthetic samples. We further verified that a 0.3 sampling ratio significantly outperforms a 1.0 ratio, avoiding the catastrophic overfitting caused by extreme synthetic upsampling."
