# Exploratory Data Analysis (EDA) Summary

## 1. Dataset Overview
- **Total Transactions Analyzed**: 284,807
- **Legitimate Transactions (Class 0)**: 284,315 (99.827%)
- **Fraudulent Transactions (Class 1)**: 492 (0.173%)
- **Class Imbalance Ratio**: ~1 fraud per 577 transactions (0.173% minority rate).
- **Missing / Null Values**: 0 (Complete clean dataset).

## 2. Amount and Time Analysis
- **Legitimate Amounts**: Mean = $55.12, Median = $22.13, Max = $7387.13
- **Fraudulent Amounts**: Mean = $169.21, Median = $42.30, Max = $7721.25
- **Observation**: Fraudulent transactions exhibit dual patterns:
  1. Low-value transactions (~$1 to $15) commonly used to verify stolen card validity.
  2. Larger sudden spikes depleting available limits before detection.
- **Time Distribution**: Spans 48 hours (172,800 seconds). Legitimate volume follows circadian peaks and nocturnal troughs, whereas fraud activity is more evenly distributed across low-traffic nocturnal hours.

## 3. Correlation & Key Predictors
- **Strongest Negative Correlates**: `['V14', 'V17', 'V12', 'V10', 'V7', 'V3']` (Values drop significantly during fraud).
- **Strongest Positive Correlates**: `['V6', 'V27', 'V25', 'Amount', 'V11', 'V4']` (Values rise sharply during fraud).
- **Preprocessing Requirement**: `Amount` and `Time` have vastly different scales compared to the normalized PCA components (V1–V28), requiring robust feature scaling (e.g., `RobustScaler`) to mitigate outlier distortion.
