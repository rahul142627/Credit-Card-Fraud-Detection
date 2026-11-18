import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS
)

def create_logistic_regression(params: dict = None) -> LogisticRegression:
    """
    Initializes a Logistic Regression classifier optimized for high-dimensional
    credit card fraud prediction.
    """
    cfg = LOGISTIC_REGRESSION_PARAMS.copy()
    if params:
        cfg.update(params)
    return LogisticRegression(**cfg)

def create_random_forest(params: dict = None) -> RandomForestClassifier:
    """
    Initializes a Random Forest classifier capable of capturing non-linear feature
    interactions and providing tree-based feature importance rankings.
    """
    cfg = RANDOM_FOREST_PARAMS.copy()
    if params:
        cfg.update(params)
    return RandomForestClassifier(**cfg)
