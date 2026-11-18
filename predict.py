import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import (
    LOGISTIC_REGRESSION_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    SCALER_PATH
)
from src.preprocess import engineer_features

def load_inference_artifacts(model_type: str = "rf"):
    """
    Loads trained model and fitted scaler for inference.
    """
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Please run main.py first.")

    scaler = joblib.load(SCALER_PATH)

    if model_type.lower() in ["rf", "random_forest"]:
        model_path = RANDOM_FOREST_MODEL_PATH
        model_name = "Random Forest"
    else:
        model_path = LOGISTIC_REGRESSION_MODEL_PATH
        model_name = "Logistic Regression"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please run main.py first.")

    model = joblib.load(model_path)
    return model, scaler, model_name

def predict_single_transaction(transaction_dict: dict, model_type: str = "rf", threshold: float = 0.5):
    """
    Infers fraud probability for an individual transaction dictionary.
    Expected fields: Time, Amount, V1..V28 (if V features are missing, defaults to 0).
    """
    model, scaler, model_name = load_inference_artifacts(model_type)

    # Build single row dataframe
    row = {}
    row["Time"] = transaction_dict.get("Time", 50000.0)
    row["Amount"] = transaction_dict.get("Amount", 50.0)

    for i in range(1, 29):
        col = f"V{i}"
        row[col] = transaction_dict.get(col, 0.0)

    df_single = pd.DataFrame([row])

    # Feature engineering
    df_single["Hour_of_Day"] = ((df_single["Time"] // 3600) % 24).astype(int)
    df_single["Log_Amount"] = np.log1p(df_single["Amount"])

    # Scale Amount and Time
    scaled_cols = scaler.transform(df_single[["Amount", "Time"]])
    df_single["scaled_amount"] = scaled_cols[:, 0]
    df_single["scaled_time"] = scaled_cols[:, 1]
    df_features = df_single.drop(columns=["Amount", "Time"])

    # Predict
    prob = model.predict_proba(df_features)[0, 1]
    is_fraud = prob >= threshold

    return {
        "model_used": model_name,
        "fraud_probability": round(float(prob), 4),
        "is_fraud": bool(is_fraud),
        "classification": "FRAUD DETECTED [ALERT]" if is_fraud else "LEGITIMATE TRANSACTION [APPROVED]",
        "risk_level": "HIGH" if prob > 0.7 else ("MEDIUM" if prob >= 0.4 else "LOW"),
        "transaction_details": {
            "Amount": f"${row['Amount']:.2f}",
            "Hour": f"{row['Time'] // 3600 % 24:.0f}:00",
            "Key_V14": row.get("V14", 0.0),
            "Key_V12": row.get("V12", 0.0),
            "Key_V4": row.get("V4", 0.0)
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Real-Time Inference CLI")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "lr"], help="Model type: rf (Random Forest) or lr (Logistic Regression)")
    parser.add_argument("--amount", type=float, default=125.50, help="Transaction amount in USD")
    parser.add_argument("--time", type=float, default=45000.0, help="Transaction time in seconds")
    parser.add_argument("--test-fraud-signature", action="store_true", help="Simulate known fraudulent PCA signature")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default 0.5)")

    args = parser.parse_args()

    tx = {"Time": args.time, "Amount": args.amount}
    if args.test_fraud_signature:
        # Inject known fraud PCA anomalies
        tx["V14"] = -7.2
        tx["V12"] = -6.5
        tx["V10"] = -5.8
        tx["V4"] = 4.8
        tx["V17"] = -6.1
        print("\n[INFO] Simulating transaction with known fraudulent signature (anomalous V14, V12, V10, V4)...")
    else:
        print("\n[INFO] Evaluating normal baseline transaction...")

    result = predict_single_transaction(tx, model_type=args.model, threshold=args.threshold)

    print("\n" + "=" * 55)
    print("        TRANSACTION FRAUD DETECTION RESULT")
    print("=" * 55)
    print(f"Model Engine:        {result['model_used']}")
    print(f"Amount:              {result['transaction_details']['Amount']}")
    print(f"Fraud Probability:   {result['fraud_probability'] * 100:.2f}%")
    print(f"Risk Assessment:     {result['risk_level']} RISK")
    print(f"Verdict:             {result['classification']}")
    print("=" * 55)

if __name__ == "__main__":
    main()
