import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_loader import load_creditcard_data
from src.eda import run_eda
from src.preprocess import preprocess_and_split
from src.smote_handler import apply_smote
from src.train import train_models
from src.evaluate import evaluate_models
from src.feature_importance import analyze_feature_importance

def run_pipeline(force_generate: bool = False, skip_eda: bool = False):
    """
    Executes the complete end-to-end Credit Card Fraud Detection pipeline:
    1. Dataset loading (284,807 transactions)
    2. Exploratory Data Analysis (EDA)
    3. Feature Engineering & Robust Scaling
    4. Stratified Train-Test Splitting
    5. Severe Class Imbalance Handling via SMOTE
    6. Classification Model Training (Logistic Regression & Random Forest)
    7. Evaluation (Precision, Recall, F1, ROC-AUC, Confusion Matrix)
    8. Feature Importance Analysis
    """
    total_start_time = time.time()
    print("=" * 70)
    print(" CREDIT CARD FRAUD DETECTION PIPELINE ")
    print(" Machine Learning with Python, Pandas, Scikit-learn & SMOTE ")
    print("=" * 70)

    # Step 1: Load Data
    df = load_creditcard_data(force_generate=force_generate)

    # Step 2: Exploratory Data Analysis
    if not skip_eda:
        run_eda(df)
    else:
        print("\n[EDA] Skipped as requested.")

    # Step 3: Feature Engineering, Scaling & Stratified Split
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(df)

    # Step 4: Handle Class Imbalance with SMOTE on Training Split
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    # Step 5: Train Models
    models = train_models(X_train_resampled, y_train_resampled)

    # Step 6: Evaluate Models on Stratified Test Data
    metrics = evaluate_models(models, X_test, y_test)

    # Step 7: Feature Importance Analysis
    feature_names = list(X_train.columns)
    importance_summary = analyze_feature_importance(models, feature_names)

    elapsed_time = time.time() - total_start_time
    print("\n" + "=" * 70)
    print(f" PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_time:.1f} SECONDS! ")
    print("=" * 70)
    print("\nArtifacts and Reports generated:")
    print("  • Saved Models:      models/logistic_regression.joblib, models/random_forest.joblib")
    print("  • Fitted Scaler:     models/scaler.joblib")
    print("  • Visual Figures:    reports/figures/*.png")
    print("  • Evaluation Report: reports/model_evaluation_report.md")
    print("  • EDA Summary:       reports/eda_summary.md")
    print("  • Metrics JSON:      reports/model_evaluation_metrics.json")
    print("=" * 70)

if __name__ == "__main__":
    force_gen = "--generate" in sys.argv
    skip_eda_flag = "--skip-eda" in sys.argv
    run_pipeline(force_generate=force_gen, skip_eda=skip_eda_flag)
