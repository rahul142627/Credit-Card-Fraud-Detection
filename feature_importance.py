import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import FIGURES_DIR

def analyze_feature_importance(models: dict, feature_names: list, top_n: int = 12, output_dir: Path = FIGURES_DIR) -> dict:
    """
    Performs comprehensive feature importance analysis:
    - Random Forest: Mean Decrease in Impurity (Gini Importance)
    - Logistic Regression: Normalized regression coefficients (Directional odds ratios)
    Saves visual comparison to disk and returns ranking dictionaries.
    """
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE & KEY FRAUD PREDICTOR ANALYSIS")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Random Forest Feature Importance
    if "Random Forest" in models:
        rf = models["Random Forest"]
        importances = rf.feature_importances_
        df_rf = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        top_rf = df_rf.head(top_n)
        summary["random_forest_top_features"] = top_rf.to_dict(orient="records")

        sns.barplot(data=top_rf, x="Importance", y="Feature", hue="Feature", ax=axes[0], palette="viridis", legend=False)
        axes[0].set_title(f"Random Forest Top {top_n} Key Fraud Predictors (MDI)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Relative Importance Score", fontsize=11)
        axes[0].set_ylabel("Engineered Feature", fontsize=11)
        for i, val in enumerate(top_rf["Importance"]):
            axes[0].text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

        print("\nTop 5 Random Forest Predictors:")
        for idx, row in top_rf.head(5).iterrows():
            print(f"  {idx + 1}. {row['Feature']}: {row['Importance']:.4f}")

    # 2. Logistic Regression Coefficients
    if "Logistic Regression" in models:
        lr = models["Logistic Regression"]
        coefs = lr.coef_[0]
        df_lr = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefs,
            "Abs_Coefficient": np.abs(coefs)
        }).sort_values(by="Abs_Coefficient", ascending=False).reset_index(drop=True)

        top_lr = df_lr.head(top_n)
        summary["logistic_regression_top_features"] = top_lr.to_dict(orient="records")

        colors = ["#e74c3c" if c > 0 else "#3498db" for c in top_lr["Coefficient"]]
        bars = axes[1].barh(top_lr["Feature"], top_lr["Coefficient"], color=colors, edgecolor="black")
        axes[1].axvline(0, color="gray", linestyle="--", alpha=0.7)
        axes[1].set_title(f"Logistic Regression Top {top_n} Feature Coefficients", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Model Coefficient (Red = Increases Fraud Odds, Blue = Decreases)", fontsize=11)
        axes[1].invert_yaxis()
        for bar in bars:
            w = bar.get_width()
            axes[1].text(w + (0.05 if w > 0 else -0.35), bar.get_y() + bar.get_height() / 2,
                         f"{w:.2f}", va="center", fontsize=9, fontweight="bold")

        print("\nTop 5 Logistic Regression Magnitude Predictors:")
        for idx, row in top_lr.head(5).iterrows():
            direction = "Positive (Fraud Indicator)" if row['Coefficient'] > 0 else "Negative (Fraud Deterrent)"
            print(f"  {idx + 1}. {row['Feature']}: {row['Coefficient']:.4f} ({direction})")

    plt.tight_layout()
    chart_path = output_dir / "feature_importance_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\nSaved feature importance comparison chart to: {chart_path.name}")

    return summary

if __name__ == "__main__":
    import joblib
    from src.config import (
        LOGISTIC_REGRESSION_MODEL_PATH,
        RANDOM_FOREST_MODEL_PATH
    )
    from src.data_loader import load_creditcard_data
    from src.preprocess import preprocess_and_split

    df = load_creditcard_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(df)
    models = {
        "Logistic Regression": joblib.load(LOGISTIC_REGRESSION_MODEL_PATH),
        "Random Forest": joblib.load(RANDOM_FOREST_MODEL_PATH)
    }
    analyze_feature_importance(models, list(X_train.columns))
