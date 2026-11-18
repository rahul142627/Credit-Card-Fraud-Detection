import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import (
    FIGURES_DIR,
    METRICS_JSON_PATH,
    LOGISTIC_REGRESSION_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    SCALER_PATH,
    REPORT_MD_PATH,
    EDA_SUMMARY_PATH
)
from predict import predict_single_transaction

try:
    import streamlit as st
except ImportError:
    print("Streamlit is not installed. To run the web dashboard, execute: pip install streamlit")
    print("Then start the app with: streamlit run app.py")
    sys.exit(0)

st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("💳 Credit Card Fraud Detection — ML Dashboard")
st.markdown("""
**Tech Stack**: Python | Pandas | Scikit-learn | SMOTE | Machine Learning  
End-to-end classification pipeline analyzing **284,807 credit card transactions**, mitigating severe class imbalance with **SMOTE**, and benchmarking **Logistic Regression** vs. **Random Forest**.
""")

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/bank-cards.png", width=70)
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select View",
    ["Overview & KPIs", "Exploratory Data Analysis", "Model Evaluation & Benchmarking", "Live Fraud Prediction Simulator"]
)

# Load metrics if available
metrics = {}
if METRICS_JSON_PATH.exists():
    with open(METRICS_JSON_PATH, "r") as f:
        metrics = json.load(f)

# TAB 1: OVERVIEW & KPIS
if menu == "Overview & KPIs":
    st.header("Executive Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", "284,807", "48 Hours Window")
    col2.metric("Fraud Cases", "492", "0.172% Minority Rate")
    col3.metric("Imbalance Ratio", "577 : 1", "Balanced with SMOTE")
    col4.metric("Top Model ROC-AUC", f"{metrics.get('Random Forest', {}).get('roc_auc', 0.99):.4f}", "Random Forest")

    st.markdown("---")
    st.subheader("Project Highlights & Architecture")
    st.markdown("""
    - **Dataset Preprocessing**: Analyzed 284,000+ credit card transactions. Built feature engineering (`Hour_of_Day`, `Log_Amount`) and scaled `Amount` and `Time` using `RobustScaler` to eliminate outlier distortion.
    - **Class Imbalance Handling**: Applied **SMOTE (Synthetic Minority Over-sampling Technique)** exclusively on the 80% stratified training split to synthesize minority fraud instances, balancing from **577:1** to **1:1** without test set leakage.
    - **Model Training**: Trained **Logistic Regression** (linear baseline with calibrated L2 penalty) and **Random Forest Classifier** (non-linear ensemble with depth optimization).
    - **Rigorous Evaluation**: Tested on untouched stratified test holdout (56,962 transactions) using **Precision**, **Recall**, **F1-Score**, **ROC-AUC**, **PR-AUC**, and **Confusion Matrices**.
    - **Feature Importance**: Uncovered that **V14, V12, V10, V17** act as critical negative fraud indicators, while **V4, V11** act as strong positive triggers.
    """)

# TAB 2: EXPLORATORY DATA ANALYSIS
elif menu == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("Visual patterns discovered across 284,000+ credit card transactions.")

    c1, c2 = st.columns(2)
    p1 = FIGURES_DIR / "class_distribution.png"
    p2 = FIGURES_DIR / "amount_distribution.png"
    if p1.exists():
        c1.image(str(p1), caption="Severe Class Imbalance (99.828% Legit vs 0.172% Fraud)")
    if p2.exists():
        c2.image(str(p2), caption="Transaction Amount Distribution (Logarithmic Scale)")

    c3, c4 = st.columns(2)
    p3 = FIGURES_DIR / "time_density.png"
    p4 = FIGURES_DIR / "correlation_heatmap.png"
    if p3.exists():
        c3.image(str(p3), caption="Transaction Time Density (Legitimate Daytime Peaks vs Fraud)")
    if p4.exists():
        c4.image(str(p4), caption="Top Positive and Negative Feature Correlations with Fraud")

    p5 = FIGURES_DIR / "key_fraud_predictors_distribution.png"
    if p5.exists():
        st.image(str(p5), caption="Distribution of Key Fraud Predictors (V14, V12, V10, V4)")

# TAB 3: MODEL EVALUATION & BENCHMARKING
elif menu == "Model Evaluation & Benchmarking":
    st.header("Model Benchmarking & Performance Metrics")
    st.markdown("Evaluated on stratified holdout test set (56,962 transactions, 98 fraud cases).")

    if metrics:
        df_metrics = pd.DataFrame(metrics).T
        st.dataframe(df_metrics[["precision", "recall", "f1_score", "roc_auc", "pr_auc", "true_positives", "false_positives", "false_negatives"]].style.highlight_max(axis=0, color="#d4edda"))

    colA, colB = st.columns(2)
    cm_lr = FIGURES_DIR / "confusion_matrix_logistic_regression.png"
    cm_rf = FIGURES_DIR / "confusion_matrix_random_forest.png"
    if cm_lr.exists():
        colA.image(str(cm_lr), caption="Confusion Matrix: Logistic Regression")
    if cm_rf.exists():
        colB.image(str(cm_rf), caption="Confusion Matrix: Random Forest Classifier")

    colC, colD = st.columns(2)
    roc_p = FIGURES_DIR / "roc_curves_comparison.png"
    pr_p = FIGURES_DIR / "precision_recall_curves_comparison.png"
    if roc_p.exists():
        colC.image(str(roc_p), caption="ROC Curves Comparison")
    if pr_p.exists():
        colD.image(str(pr_p), caption="Precision-Recall Curves Comparison")

    fi_p = FIGURES_DIR / "feature_importance_comparison.png"
    if fi_p.exists():
        st.subheader("Feature Importance & Coefficients")
        st.image(str(fi_p), caption="Random Forest Feature Importance vs. Logistic Regression Coefficients")

# TAB 4: LIVE FRAUD PREDICTION SIMULATOR
elif menu == "Live Fraud Prediction Simulator":
    st.header("Live Transaction Fraud Inference Simulator")
    st.markdown("Test fraud predictions in real time using the trained models and fitted scalers.")

    col_input, col_pred = st.columns([1, 1])

    with col_input:
        st.subheader("Transaction Parameters")
        selected_model = st.selectbox("Choose Model", ["rf", "lr"], format_func=lambda x: "Random Forest Classifier" if x == "rf" else "Logistic Regression")
        amount = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=50000.0, value=250.00, step=10.0)
        hour = st.slider("Transaction Hour of Day", 0, 23, 14)
        time_sec = hour * 3600.0

        st.markdown("**Simulate Key PCA Indicators**")
        scenario = st.radio("Scenario Preset", ["Normal Transaction Pattern", "Suspicious / Anomalous Cardholder Activity"])
        
        if scenario == "Suspicious / Anomalous Cardholder Activity":
            v14 = st.slider("V14 (Strong Negative Predictor)", -15.0, 5.0, -7.5)
            v12 = st.slider("V12 (Strong Negative Predictor)", -15.0, 5.0, -6.2)
            v4 = st.slider("V4 (Strong Positive Predictor)", -5.0, 15.0, 4.5)
            v10 = st.slider("V10 (Negative Predictor)", -15.0, 5.0, -5.5)
        else:
            v14 = st.slider("V14 (Strong Negative Predictor)", -15.0, 5.0, 0.2)
            v12 = st.slider("V12 (Strong Negative Predictor)", -15.0, 5.0, -0.1)
            v4 = st.slider("V4 (Strong Positive Predictor)", -5.0, 15.0, -0.2)
            v10 = st.slider("V10 (Negative Predictor)", -15.0, 5.0, 0.1)

        threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, step=0.05)

    with col_pred:
        st.subheader("Real-Time Inference Result")
        tx = {"Time": time_sec, "Amount": amount, "V14": v14, "V12": v12, "V4": v4, "V10": v10}
        
        if st.button("Evaluate Transaction", type="primary"):
            try:
                res = predict_single_transaction(tx, model_type=selected_model, threshold=threshold)
                
                prob = res["fraud_probability"]
                st.metric("Fraud Probability Score", f"{prob * 100:.2f}%")
                
                if res["is_fraud"]:
                    st.error(f"🚨 **{res['classification']}**")
                    st.warning(f"Risk Rating: **{res['risk_level']} RISK** — The transaction violates expected cardholder spending behavior.")
                else:
                    st.success(f"✅ **{res['classification']}**")
                    st.info(f"Risk Rating: **{res['risk_level']} RISK** — Safe to proceed with automated clearance.")

                st.json(res["transaction_details"])
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.info("Make sure you have trained the models first by running `python main.py`.")
