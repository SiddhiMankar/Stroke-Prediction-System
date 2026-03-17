# Phase 3: Threshold Optimization

## Objective
To incrementally adjust and optimize the prediction probability threshold of the selected model (`64 → 32 → 16`) to maximize its ability to detect stroke patients (Recall) while understanding the trade-off with false positives (Precision).

## Methodology
- **Approach:** Instead of relying on the default binary classification threshold (`0.5`), the continuous probability scores (`0.0` to `1.0`) outputted by the model were extracted.
- **Thresholds Evaluated:** The model's predictions were re-calculated across a range of thresholds: `0.1` through `0.9`. 
    - E.g., if threshold = `0.3`, any patient with a stroke probability score ≥ `30%` is flagged as a positive stroke case.
- **Metrics Tracked:** Recall (sensitivity/true positive detection rate) and Precision (positive predictive value).

## Results: The Trade-off
The performance metrics across different thresholds on the validation set reveal the expected inverse relationship between Recall and Precision.

| Threshold | Recall | Precision | False Negatives (Missed Strokes) | True Positives (Detected) |
| :--- | :--- | :--- | :--- | :--- |
| **0.1** | 0.903 | 0.128 | 6 | 56 |
| **0.2** | 0.774 | 0.158 | 14 | 48 |
| **0.3** | **0.677** | **0.180** | **20** | **42** |
| **0.4** | **0.629** | **0.217** | **23** | **39** |
| **0.5 (Default)** | 0.516 | 0.224 | 30 | 32 |
| **0.6** | 0.435 | 0.252 | 35 | 27 |
| **0.7** | 0.387 | 0.270 | 38 | 24 |
| **0.8** | 0.242 | 0.241 | 47 | 15 |
| **0.9** | 0.113 | 0.226 | 55 | 7 |

*Note: The test set contained 62 actual stroke cases.*

## Interpretation & Selection

**The Core Trend:** As the probability threshold decreases, the model becomes less strict about making a "stroke" prediction. Consequently, it successfully catches more true stroke patients (Recall increases). However, it also incorrectly flags more healthy patients as having a stroke (Precision decreases).

**The Clinical Justification:** In medical diagnostics, particularly for a severe and time-sensitive disease like a stroke, minimizing **False Negatives** is the absolute highest priority. A False Negative means a patient actually had a stroke, but the model gave them an "all clear", which is extremely dangerous. A False Positive simply results in a patient undergoing further, potentially unnecessary, preventative medical screening. 

**Conclusion:** 
Relying on the default `0.5` threshold is insufficient as the model misses nearly half of the stroke cases (30 false negatives out of 62). 

By optimizing the threshold down to **`0.3`** or **`0.4`**, we significantly improve the model's clinical utility.
- At **`0.3`**, we achieve a strong **67.7% Recall**, detecting 10 more strokes than the default threshold, making it the mathematically safer choice for a medical screening tool. 
- At **`0.4`**, we achieve a **62.9% Recall** with a slightly better precision balance.

The threshold **`0.3`** has been identified as the optimal point that prioritizes patient safety by aggressively reducing false negatives while keeping precision within acceptable bounds for a highly imbalanced dataset. A visual representation of this trade-off has been generated in `threshold_tradeoff.png`.
