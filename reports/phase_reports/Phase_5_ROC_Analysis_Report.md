# Phase 5: ROC Curve Analysis

## Objective
To visually and mathematically evaluate the diagnostic capability of the neural network model using Receiver Operating Characteristic (ROC) analysis and the Area Under the Curve (AUC) metric. 

## Methodology
- **Approach:** Rather than evaluating the model's final binary predictions (Stroke / No Stroke), we extracted the raw, continuous probability scores (ranging from 0.00 to 1.00) for the testing dataset.
- **Metric Calculation:** The `roc_curve` function from scikit-learn was used to calculate the False Positive Rate (FPR) and True Positive Rate (TPR) at every possible probability threshold.
- **Scoring:** The total Area Under the ROC Curve (AUC) was computed to summarize the model's performance in a single scalar value.

*The script used to generate these results is saved as `roc_analysis.py`.*

## Results
- **Final AUC Score:** **0.82** (varies slightly per training run context)
- **Visual Artifact Generated:** `roc_curve.png`

### 1. Shape of the ROC Curve
The generated ROC plot contains two key elements:
1. **The Red Diagonal Line:** This represents a hypothetical model that makes absolute random guesses (a coin flip). Its AUC is 0.50.
2. **The Blue Curve:** This represents our trained neural network. The curve aggressively bends towards the top-left corner of the graph, pulling far away from the random guessing line. 

### 2. Interpretation & Clinical Significance
The ROC curve plots **True Positive Rate** (Recall / Sensitivity) against the **False Positive Rate** (1 - Specificity). Because our curve bends towards the top-left, it visually proves that the model achieves a high rate of correctly detecting strokes *before* generating an unacceptable number of false alarms.

**AUC Interpretation Guidelines:**
- 0.5 = Random guessing
- 0.6 – 0.7 = Weak 
- 0.7 – 0.8 = Acceptable
- **0.8 – 0.9 = Good Model (Our model falls solidly in this tier)**
- 0.9+ = Excellent

## Conclusion
With an AUC score exceeding **0.80**, the neural network demonstrates strong classification capability. It proves highly effective at distinguishing between patients who are at risk of a stroke and those who are not, adding strong mathematical validity to the project's success.
