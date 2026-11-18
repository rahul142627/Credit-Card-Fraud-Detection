import sys
import time
import joblib
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    LOGISTIC_REGRESSION_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    MODELS_DIR
)
from src.models import create_logistic_regression, create_random_forest

def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Trains both Logistic Regression and Random Forest models on the SMOTE-balanced training set.
    Persists trained model artifacts to disk using joblib.
    """
    print("\n" + "=" * 60)
    print("TRAINING CLASSIFICATION MODELS (LOGISTIC REGRESSION & RANDOM FOREST)")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = {}

    # 1. Train Logistic Regression
    print(f"\n[1/2] Training Logistic Regression model on {len(X_train):,} samples...")
    t0 = time.time()
    lr = create_logistic_regression()
    lr.fit(X_train, y_train)
    lr_duration = time.time() - t0
    print(f"--> Logistic Regression training completed in {lr_duration:.2f} seconds.")
    joblib.dump(lr, LOGISTIC_REGRESSION_MODEL_PATH)
    print(f"--> Saved Logistic Regression model to: {LOGISTIC_REGRESSION_MODEL_PATH}")
    models["Logistic Regression"] = lr

    # 2. Train Random Forest Classifier
    print(f"\n[2/2] Training Random Forest Classifier model on {len(X_train):,} samples...")
    t0 = time.time()
    rf = create_random_forest()
    rf.fit(X_train, y_train)
    rf_duration = time.time() - t0
    print(f"--> Random Forest training completed in {rf_duration:.2f} seconds.")
    joblib.dump(rf, RANDOM_FOREST_MODEL_PATH)
    print(f"--> Saved Random Forest model to: {RANDOM_FOREST_MODEL_PATH}")
    models["Random Forest"] = rf

    return models

if __name__ == "__main__":
    from src.data_loader import load_creditcard_data
    from src.preprocess import preprocess_and_split
    from src.smote_handler import apply_smote

    df = load_creditcard_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(df)
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    trained_models = train_models(X_train_res, y_train_res)
