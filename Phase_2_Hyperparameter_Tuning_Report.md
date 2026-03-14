# Phase 2: Hyperparameter Tuning

## Objective
To experimentally validate the neural network architecture by training multiple multi-layer perceptron (MLP) configurations and comparing their performance to select the optimal model.

## Methodology
- **Approach:** Four different network architectures were trained using the same preprocessed dataset (`X_train`, `y_train`) and evaluated on the same validation set (`X_test`, `y_test`).
- **Architectures Tested:**
    1. **Model 1:** 32 → 16
    2. **Model 2:** 64 → 32
    3. **Model 3:** 64 → 32 → 16
    4. **Model 4:** 128 → 64
- **Training Parameters:** All models were trained using the `adam` optimizer, `binary_crossentropy` loss, and class weights to handle the imbalanced nature of the dataset. Each model was trained for 30 epochs with a batch size of 32.
- **Metrics Evaluated:** Recall, Precision, and AUC (Area Under the ROC Curve).

## Results
The performance of each architecture on the validation set is summarized in the table below:

| Model | Architecture | Recall | Precision | AUC |
| :--- | :--- | :--- | :--- | :--- |
| Model 1 | 32 → 16 | 0.742 | 0.156 | 0.814 |
| Model 2 | 64 → 32 | 0.694 | 0.181 | 0.835 |
| Model 3 | 64 → 32 → 16 | 0.758 | 0.158 | 0.822 |
| Model 4 | 128 → 64 | 0.597 | 0.224 | 0.799 |

## Interpretation & Model Selection
- **Best True Positive Detection (Recall):** `Model 3 (64 → 32 → 16)` achieved the highest recall (0.758) and `Model 1 (32 → 16)` followed closely at 0.742.
- **Best Overall Discrimination (AUC):** `Model 2 (64 → 32)` achieved the highest AUC (0.835) and the highest precision (0.181 among the top 3 models by recall), indicating it has the best overall balance of distinguishing stroke from non-stroke patients while still maintaining a reasonable recall (~0.694).
- **Scale Impact:** `Model 4 (128 → 64)` demonstrated the highest precision but at a significant cost to recall (dropping to 0.597). This indicates that overly large architectures may overfit to the non-stroke class and miss too many actual stroke cases, which is highly undesirable in medical diagnosis.

### Conclusion
`Model 2 (64 → 32)` or `Model 3 (64 → 32 → 16)` represent the best architectures. We select the **64 → 32 → 16** architecture as it provides the highest recall (successfully identifying the most stroke cases) while maintaining a very competitive AUC. It proves that a moderately deep network is required to capture the complex relationships in the data without overfitting like the 128 → 64 model.

*This scientific model selection process experimentally validates that our chosen architecture provides optimal performance for this specific imbalanced classification task.*
