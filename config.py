import os
import tempfile
from pathlib import Path

# Silence loky and matplotlib cache issues on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure runtime directories exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File Paths
DATA_CSV_PATH = DATA_DIR / "creditcard.csv"
LOGISTIC_REGRESSION_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METRICS_JSON_PATH = REPORTS_DIR / "model_evaluation_metrics.json"
REPORT_MD_PATH = REPORTS_DIR / "model_evaluation_report.md"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.md"

# Dataset Specs (Matching Kaggle Credit Card Fraud benchmark)
TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.20
EXPECTED_NUM_ROWS = 284807
EXPECTED_FRAUD_COUNT = 492

# SMOTE Configuration
SMOTE_K_NEIGHBORS = 5
SMOTE_SAMPLING_STRATEGY = "auto"

# Hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    "C": 0.1,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
    "class_weight": None,  # Handled by SMOTE in pipeline
    "n_jobs": 1
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 14,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": 2,
    "verbose": 0
}
