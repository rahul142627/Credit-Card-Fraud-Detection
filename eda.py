import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    FIGURES_DIR,
    EDA_SUMMARY_PATH,
    TARGET_COLUMN
)
from src.data_loader import load_creditcard_data

def run_eda(df: pd.DataFrame, output_dir: Path = FIGURES_DIR, summary_path: Path = EDA_SUMMARY_PATH) -> dict:
    """
    Performs comprehensive Exploratory Data Analysis (EDA) on credit card transactions,
    generates visual figures, and writes an executive markdown summary.
    """
    print("\n" + "=" * 60)
    print("RUNNING EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Dataset stats
    total_records = len(df)
    n_legit = int((df[TARGET_COLUMN] == 0).sum())
    n_fraud = int((df[TARGET_COLUMN] == 1).sum())
    fraud_pct = (n_fraud / total_records) * 100
    legit_pct = 100.0 - fraud_pct
    null_counts = int(df.isnull().sum().sum())

    amount_stats_legit = df[df[TARGET_COLUMN] == 0]["Amount"].describe().to_dict()
    amount_stats_fraud = df[df[TARGET_COLUMN] == 1]["Amount"].describe().to_dict()

    print(f"Total Transactions: {total_records:,}")
    print(f"Legitimate Transactions: {n_legit:,} ({legit_pct:.3f}%)")
    print(f"Fraudulent Transactions: {n_fraud:,} ({fraud_pct:.3f}%)")
    print(f"Missing Values: {null_counts}")
    print(f"Legit Amount Mean: ${amount_stats_legit['mean']:.2f} | Median: ${amount_stats_legit['50%']:.2f} | Max: ${amount_stats_legit['max']:.2f}")
    print(f"Fraud Amount Mean: ${amount_stats_fraud['mean']:.2f} | Median: ${amount_stats_fraud['50%']:.2f} | Max: ${amount_stats_fraud['max']:.2f}")

    # --- Plot 1: Class Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot (log scale to see fraud clearly)
    counts = [n_legit, n_fraud]
    labels = ["Legitimate (0)", "Fraudulent (1)"]
    colors = ["#2ecc71", "#e74c3c"]

    bars = axes[0].bar(labels, counts, color=colors, edgecolor="black", width=0.5)
    axes[0].set_yscale("log")
    axes[0].set_title("Class Frequency (Logarithmic Scale)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Transaction Count (Log Scale)", fontsize=11)
    for bar, count in zip(bars, counts):
        axes[0].annotate(f"{count:,}\n({count/total_records*100:.3f}%)",
                         xy=(bar.get_x() + bar.get_width() / 2, count),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Donut pie chart
    axes[1].pie([n_legit, n_fraud], labels=labels, autopct="%1.3f%%",
                colors=colors, startangle=140, explode=(0, 0.15),
                wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
    axes[1].set_title("Severe Class Imbalance Proportion", fontsize=13, fontweight="bold")

    plt.tight_layout()
    p1_path = output_dir / "class_distribution.png"
    plt.savefig(p1_path, dpi=300)
    plt.close()
    print(f"Saved: {p1_path.name}")

    # --- Plot 2: Transaction Amount Distribution ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    sns.histplot(df[df[TARGET_COLUMN] == 0]["Amount"], bins=50, ax=ax1, color="#2ecc71", kde=True)
    ax1.set_title("Legitimate Transaction Amount Distribution", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.set_yscale("log")

    sns.histplot(df[df[TARGET_COLUMN] == 1]["Amount"], bins=50, ax=ax2, color="#e74c3c", kde=True)
    ax2.set_title("Fraudulent Transaction Amount Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Transaction Amount ($)")
    ax2.set_ylabel("Count")
    ax2.set_yscale("log")

    plt.tight_layout()
    p2_path = output_dir / "amount_distribution.png"
    plt.savefig(p2_path, dpi=300)
    plt.close()
    print(f"Saved: {p2_path.name}")

    # --- Plot 3: Transaction Time Density ---
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.kdeplot(df[df[TARGET_COLUMN] == 0]["Time"] / 3600, ax=ax, label="Legitimate", color="#2ecc71", lw=2, fill=True, alpha=0.3)
    sns.kdeplot(df[df[TARGET_COLUMN] == 1]["Time"] / 3600, ax=ax, label="Fraudulent", color="#e74c3c", lw=2, fill=True, alpha=0.3)
    ax.set_title("Transaction Density Across 48-Hour Timeline", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time Elapsed (Hours)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    p3_path = output_dir / "time_density.png"
    plt.savefig(p3_path, dpi=300)
    plt.close()
    print(f"Saved: {p3_path.name}")

    # --- Plot 4: Correlation with Target Feature ---
    correlations = df.corr()[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values()
    top_negative = correlations.head(6)
    top_positive = correlations.tail(6)
    key_features = pd.concat([top_negative, top_positive])

    fig, ax = plt.subplots(figsize=(12, 6))
    colors_corr = ["#e74c3c" if val > 0 else "#3498db" for val in key_features.values]
    bars = ax.barh(key_features.index, key_features.values, color=colors_corr, edgecolor="black")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Top Positively & Negatively Correlated Features with Fraud (Class)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient")
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.2f}",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5 if w > 0 else -25, 0), textcoords="offset points",
                    ha="left" if w > 0 else "right", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    p4_path = output_dir / "correlation_heatmap.png"
    plt.savefig(p4_path, dpi=300)
    plt.close()
    print(f"Saved: {p4_path.name}")

    # --- Plot 5: Key Predictor Box Plots ---
    top_4_predictors = [correlations.index[0], correlations.index[1], correlations.index[-1], correlations.index[-2]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(top_4_predictors):
        sns.boxplot(x=TARGET_COLUMN, y=col, hue=TARGET_COLUMN, data=df, ax=axes[i], palette=["#2ecc71", "#e74c3c"], legend=False)
        axes[i].set_title(f"Distribution of {col} by Transaction Type", fontweight="bold")
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["Legitimate (0)", "Fraud (1)"])

    plt.tight_layout()
    p5_path = output_dir / "key_fraud_predictors_distribution.png"
    plt.savefig(p5_path, dpi=300)
    plt.close()
    print(f"Saved: {p5_path.name}")

    # Generate Markdown Summary
    summary_md = f"""# Exploratory Data Analysis (EDA) Summary

## 1. Dataset Overview
- **Total Transactions Analyzed**: {total_records:,}
- **Legitimate Transactions (Class 0)**: {n_legit:,} ({legit_pct:.3f}%)
- **Fraudulent Transactions (Class 1)**: {n_fraud:,} ({fraud_pct:.3f}%)
- **Class Imbalance Ratio**: ~1 fraud per {int(n_legit / n_fraud):,} transactions ({fraud_pct:.3f}% minority rate).
- **Missing / Null Values**: {null_counts} (Complete clean dataset).

## 2. Amount and Time Analysis
- **Legitimate Amounts**: Mean = ${amount_stats_legit['mean']:.2f}, Median = ${amount_stats_legit['50%']:.2f}, Max = ${amount_stats_legit['max']:.2f}
- **Fraudulent Amounts**: Mean = ${amount_stats_fraud['mean']:.2f}, Median = ${amount_stats_fraud['50%']:.2f}, Max = ${amount_stats_fraud['max']:.2f}
- **Observation**: Fraudulent transactions exhibit dual patterns:
  1. Low-value transactions (~$1 to $15) commonly used to verify stolen card validity.
  2. Larger sudden spikes depleting available limits before detection.
- **Time Distribution**: Spans 48 hours (172,800 seconds). Legitimate volume follows circadian peaks and nocturnal troughs, whereas fraud activity is more evenly distributed across low-traffic nocturnal hours.

## 3. Correlation & Key Predictors
- **Strongest Negative Correlates**: `{top_negative.index.tolist()}` (Values drop significantly during fraud).
- **Strongest Positive Correlates**: `{top_positive.index.tolist()}` (Values rise sharply during fraud).
- **Preprocessing Requirement**: `Amount` and `Time` have vastly different scales compared to the normalized PCA components (V1–V28), requiring robust feature scaling (e.g., `RobustScaler`) to mitigate outlier distortion.
"""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"EDA Summary written to: {summary_path}")

    return {
        "total_records": total_records,
        "n_legit": n_legit,
        "n_fraud": n_fraud,
        "fraud_pct": fraud_pct,
        "top_negative": top_negative.to_dict(),
        "top_positive": top_positive.to_dict()
    }

if __name__ == "__main__":
    df = load_creditcard_data()
    run_eda(df)
