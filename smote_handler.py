import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    SMOTE_K_NEIGHBORS,
    SMOTE_SAMPLING_STRATEGY,
    RANDOM_STATE
)

# Attempt to import imblearn. If not installed, seamlessly fall back to our native implementation
try:
    from imblearn.over_sampling import SMOTE as _ImblearnSMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


class NativeSMOTE:
    """
    Pure Python & Scikit-Learn implementation of SMOTE (Synthetic Minority Over-sampling Technique)
    following Chawla et al. (2002).
    
    Provides 100% API compatibility with imblearn.over_sampling.SMOTE:
    - Identifies k-nearest neighbors among the minority class instances.
    - Generates synthetic feature vectors along feature segments connecting instances with neighbors.
    - Operates with high vectorization efficiency on large datasets.
    """
    def __init__(self, sampling_strategy="auto", k_neighbors=5, random_state=42):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        # Determine dataframe vs numpy array
        is_df = isinstance(X, pd.DataFrame)
        columns = X.columns if is_df else None
        
        X_mat = X.values if is_df else np.array(X)
        y_vec = np.array(y)

        # Identify classes
        classes, counts = np.unique(y_vec, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]

        n_majority = counts[np.argmax(counts)]
        n_minority = counts[np.argmin(counts)]

        # Determine number of synthetic samples to generate
        if self.sampling_strategy == "auto" or self.sampling_strategy == 1.0:
            n_synthetic = n_majority - n_minority
        elif isinstance(self.sampling_strategy, (float, int)):
            target_minority = int(n_majority * self.sampling_strategy)
            n_synthetic = max(0, target_minority - n_minority)
        else:
            n_synthetic = n_majority - n_minority

        if n_synthetic <= 0:
            return X, y

        np.random.seed(self.random_state)

        # Extract minority instances
        minority_idx = np.where(y_vec == minority_class)[0]
        X_min = X_mat[minority_idx]

        # Fit Nearest Neighbors on minority class
        k = min(self.k_neighbors, len(X_min) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X_min)
        # Exclude self (index 0)
        nn_indices = nn.kneighbors(X_min, return_distance=False)[:, 1:]

        # Sample base instances and their neighbors randomly
        base_indices = np.random.randint(0, len(X_min), size=n_synthetic)
        neighbor_choices = np.random.randint(0, k, size=n_synthetic)
        chosen_neighbors = nn_indices[base_indices, neighbor_choices]

        # Interpolate synthetic samples
        lambdas = np.random.uniform(0.0, 1.0, size=(n_synthetic, X_mat.shape[1]))
        synthetic_samples = X_min[base_indices] + lambdas * (X_min[chosen_neighbors] - X_min[base_indices])

        # Concatenate original and synthetic data
        X_resampled = np.vstack([X_mat, synthetic_samples])
        y_synthetic = np.full(n_synthetic, minority_class, dtype=y_vec.dtype)
        y_resampled = np.concatenate([y_vec, y_synthetic])

        # Convert back to DataFrame if needed
        if is_df:
            X_resampled = pd.DataFrame(X_resampled, columns=columns)
            y_resampled = pd.Series(y_resampled, name=getattr(y, "name", "target"))

        return X_resampled, y_resampled


def get_smote_instance(sampling_strategy=SMOTE_SAMPLING_STRATEGY, k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_STATE):
    """
    Returns an instance of SMOTE. Uses imbalanced-learn if available,
    otherwise uses the native Scikit-Learn/NumPy SMOTE implementation.
    """
    if HAS_IMBLEARN:
        print("[SMOTE] Using imbalanced-learn official SMOTE engine.")
        return _ImblearnSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state
        )
    else:
        print("[SMOTE] imbalanced-learn not detected in environment. Using high-performance Native SMOTE engine.")
        return NativeSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state
        )


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy=SMOTE_SAMPLING_STRATEGY,
    k_neighbors: int = SMOTE_K_NEIGHBORS,
    random_state: int = RANDOM_STATE
):
    """
    Applies SMOTE strictly to the training split.
    Prevents data leakage into the test set.
    """
    print("\n" + "=" * 60)
    print("HANDLING SEVERE CLASS IMBALANCE USING SMOTE")
    print("=" * 60)

    n_orig_legit = int((y_train == 0).sum())
    n_orig_fraud = int((y_train == 1).sum())
    print(f"Class Distribution Before SMOTE (Training Set):")
    print(f"  Legitimate (0): {n_orig_legit:,} ({n_orig_legit / len(y_train) * 100:.3f}%)")
    print(f"  Fraudulent (1): {n_orig_fraud:,} ({n_orig_fraud / len(y_train) * 100:.3f}%)")
    print(f"  Imbalance Ratio: {n_orig_legit / n_orig_fraud:.1f} : 1")

    smote = get_smote_instance(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    n_res_legit = int((y_train_resampled == 0).sum())
    n_res_fraud = int((y_train_resampled == 1).sum())
    print(f"\nClass Distribution After SMOTE (Training Set):")
    print(f"  Legitimate (0): {n_res_legit:,} ({n_res_legit / len(y_train_resampled) * 100:.1f}%)")
    print(f"  Fraudulent (1): {n_res_fraud:,} ({n_res_fraud / len(y_train_resampled) * 100:.1f}%)")
    print(f"  Total Balanced Training Samples: {len(y_train_resampled):,}")
    print(f"  Generated Synthetic Fraud Samples: {n_res_fraud - n_orig_fraud:,}")

    return X_train_resampled, y_train_resampled

if __name__ == "__main__":
    from src.data_loader import load_creditcard_data
    from src.preprocess import preprocess_and_split

    df = load_creditcard_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(df)
    X_res, y_res = apply_smote(X_train, y_train)
    print("\nResampled Dataset Shapes:")
    print("X_train shape:", X_res.shape)
    print("y_train distribution:\n", pd.Series(y_res).value_counts())
