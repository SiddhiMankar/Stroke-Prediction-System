# Stroke EDA - Key Insights Summary

## 📊 Dataset Overview
- **Records:** 5,110
- **Target:** stroke (0 = No Stroke, 1 = Stroke)
- **Class Distribution:** ~95% no stroke, ~5% stroke

## 🎯 Critical Finding: Severe Class Imbalance
- **Implication:** Accuracy is misleading - can achieve ~95% by always predicting "no stroke"
- **Solution:** Focus on RECALL (sensitivity) in modeling
- **Recommendation:** Use SMOTE, class weights, or ensemble methods

## 🔝 Top Predictors (Ranked)

### Strong Predictors ✅
1. **Age** - Strongest correlation, clear separation in distributions
2. **Hypertension** - 2-3x higher stroke rate when present
3. **Heart Disease** - 2-3x higher stroke rate when present

### Moderate Predictors ⚠️
4. **Glucose Level** - Higher in stroke patients, moderate correlation
5. **Smoking Status** - Moderate association, "formerly smoked" highest risk
6. **Age Group** (engineered) - Clear risk bands, 60+ shows sharp increase

### Weak/Questionable Predictors ❌
7. **BMI** - Weak correlation, high overlap between groups
8. **Work Type** - Likely age-confounded
9. **Ever Married** - Age proxy (married = older)
10. **Residence Type** - Minimal impact, negligible difference
11. **Gender** - Slight difference, marginal effect

## 📈 All Visualizations Generated

✅ 15 plots saved to `eda/plots/`:
1. Stroke distribution (class imbalance)
2. Age vs stroke (boxplot + violin)
3. Glucose vs stroke (boxplot + violin)
4. BMI vs stroke (boxplot + violin)
5. Hypertension vs stroke % (percentage bar)
6. Heart disease vs stroke % (percentage bar)
7. Smoking status vs stroke % (percentage bar)
8. Work type vs stroke % (percentage bar)
9. Residence type vs stroke % (percentage bar)
10. Gender vs stroke % (percentage bar)
11. Ever married vs stroke % (percentage bar)
12. Age group vs stroke % (percentage bar)
13. BMI category vs stroke % (percentage bar)
14. Correlation heatmap (numeric features)
15. Pairplot (advanced insights)

## 🔬 Medical Validation
✅ Results align with medical knowledge:
- Age, hypertension, heart disease are known stroke risk factors
- Dataset appears reliable for predictive modeling

## 🚀 Modeling Recommendations

1. **Handle Class Imbalance:**
   - Use SMOTE or ADASYN for oversampling
   - Apply class weights in model
   - Consider ensemble methods (balanced random forest)

2. **Feature Selection:**
   - Prioritize: age, hypertension, heart_disease, avg_glucose_level
   - Consider: smoking_status, age_group
   - May drop: Residence_type (minimal impact)

3. **Evaluation Metrics:**
   - Primary: Recall (sensitivity)
   - Secondary: F1-score, AUC-ROC
   - Avoid: Accuracy (misleading due to imbalance)

4. **Feature Engineering:**
   - Age groups work well (clear risk bands)
   - Consider age × glucose interaction
   - BMI categories slightly better than continuous BMI

5. **Data Splitting:**
   - Use stratified split to maintain class balance
   - Consider age-based stratification

## 📁 Files Created
- `eda/comprehensive_eda.py` - Main analysis script
- `eda/plots/` - 15 visualization files
- Complete walkthrough with detailed insights
