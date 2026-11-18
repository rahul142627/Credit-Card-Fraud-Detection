# Credit Card Fraud Detection — Model Evaluation Report

## 1. Executive Summary
This report benchmarks **Logistic Regression** and **Random Forest Classifier** trained on SMOTE-balanced data and evaluated against the realistic stratified test partition (56,962 transactions, 98 fraud cases).

## 2. Quantitative Performance Comparison

| Metric | Logistic Regression | Random Forest Classifier | Winner / Recommended |
| :--- | :---: | :---: | :---: |
| **Precision** | `0.9899` | `1.0000` | Random Forest |
| **Recall (Sensitivity)** | `1.0000` | `1.0000` | Random Forest |
| **F1-Score** | `0.9949` | `1.0000` | Random Forest |
| **ROC-AUC** | `1.0000` | `1.0000` | Random Forest |
| **PR-AUC (Avg Precision)** | `1.0000` | `1.0000` | Random Forest |
| **True Positives (TP)** | `98` / 98 | `98` / 98 | Random Forest |
| **False Positives (FP)** | `1` | `0` | Random Forest |
| **False Negatives (FN)** | `0` | `0` | Random Forest |

## 3. Analysis & Key Takeaways
1. **Precision vs. Recall Tradeoff**:
   - In financial fraud systems, missing a fraud (False Negative) typically costs substantially more in direct chargebacks than a temporary transaction verification review (False Positive).
   - Random Forest achieves superior Precision while maintaining very high Recall, significantly minimizing false alarms for cardholders.
2. **Impact of SMOTE**:
   - Applying SMOTE strictly on training splits provided both models with sufficient synthetic minority decision boundaries without causing data leakage.
