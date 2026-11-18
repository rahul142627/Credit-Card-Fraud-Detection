import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    TEST_SIZE,
    RANDOM_STATE,
    TARGET_COLUMN,
    SCALER_PATH
)
from src.data_loader import load_creditcard_data

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs domain-specific feature engineering on credit card transactions:
    1. Hour_of_Day: Cyclical temporal feature (0-23 hours)
    2. Log_Amount: Logarithmic transformation of transaction amount to tame right-skew
    """
    df = df.copy()

    # Time engineering: 172,800s corresponds to 48 hours (2 days)
    # Extract hour of day (0 to 23)
    df["Hour_of_Day"] = ((df["Time"] // 3600) % 24).astype(int)

    # Amount engineering: log(1 + amount) handles extreme financial spikes gracefully
    df["Log_Amount"] = np.log1p(df["Amount"])

    return df

def preprocess_and_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    scaler_save_path: Path = SCALER_PATH
):
    """
    Preprocesses data:
    - Applies feature engineering
    - Stratified train-test split (80-20) preserving rare fraud ratio
    - RobustScaler applied to Amount and Time to prevent outlier distortion
    - Strictly fits scaler on training set to prevent data leakage
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)

    # 1. Feature Engineering
    df_engineered = engineer_features(df)
    print(f"Applied Feature Engineering: Added 'Hour_of_Day' and 'Log_Amount'.")

    # 2. Separate Features (X) and Target (y)
    X = df_engineered.drop(columns=[TARGET_COLUMN])
    y = df_engineered[TARGET_COLUMN]

    # 3. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Stratified Train-Test Split ({(1 - test_size) * 100:.0f}% Train / {test_size * 100:.0f}% Test):")
    print(f"  Training Set: {len(X_train):,} samples | Fraud count: {int(y_train.sum()):,} ({y_train.mean() * 100:.3f}%)")
    print(f"  Testing Set:  {len(X_test):,} samples | Fraud count: {int(y_test.sum()):,} ({y_test.mean() * 100:.3f}%)")

    # 4. Feature Scaling (RobustScaler on Amount and Time)
    features_to_scale = ["Amount", "Time"]
    scaler = RobustScaler()

    # Fit ONLY on training data to avoid data snooping
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    scaled_train_cols = scaler.fit_transform(X_train[features_to_scale])
    scaled_test_cols = scaler.transform(X_test[features_to_scale])

    X_train_scaled["scaled_amount"] = scaled_train_cols[:, 0]
    X_train_scaled["scaled_time"] = scaled_train_cols[:, 1]
    X_test_scaled["scaled_amount"] = scaled_test_cols[:, 0]
    X_test_scaled["scaled_time"] = scaled_test_cols[:, 1]

    # Drop raw unscaled Time and Amount
    X_train_scaled = X_train_scaled.drop(columns=features_to_scale)
    X_test_scaled = X_test_scaled.drop(columns=features_to_scale)

    # Save fitted scaler
    scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    print(f"RobustScaler fitted on training split and saved to {scaler_save_path}")
    print(f"Final feature dimensions: X_train {X_train_scaled.shape}, X_test {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    df = load_creditcard_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(df)
    print("\nPreprocessed Feature Columns:")
    print(list(X_train.columns))
