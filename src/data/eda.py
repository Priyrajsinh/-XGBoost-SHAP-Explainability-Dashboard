"""Exploratory data analysis — generates figures for reports/figures/.

Run via:
    python -m src.data.eda

Figures produced
----------------
reports/figures/class_dist.png
reports/figures/correlation.png
reports/figures/feature_distributions.png  (before vs after imputation)
"""

import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset import load_pima
from src.data.preprocessing import _FEATURE_COLS, impute_and_split
from src.logger import get_logger

matplotlib.use("Agg")  # headless rendering

logger = get_logger(__name__)

_FIGURES_DIR = pathlib.Path("reports/figures")


def _save(fig: plt.Figure, name: str) -> None:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = _FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved", extra={"path": str(path)})


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Bar chart of Outcome class counts."""
    counts = df["Outcome"].value_counts().sort_index()
    labels = ["Non-Diabetic (0)", "Diabetic (1)"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        labels, counts.values, color=["#4c72b0", "#dd8452"], edgecolor="black"
    )
    ax.bar_label(bars, padding=3)
    ax.set_title("Class Distribution — Pima Diabetes")
    ax.set_ylabel("Count")
    _save(fig, "class_dist.png")


def plot_correlation(df: pd.DataFrame) -> None:
    """Heatmap of Pearson correlation (NaN filled with 0 for display)."""
    corr = df.fillna(0).corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    cols = list(corr.columns)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    # annotate cells
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6
            )
    ax.set_title("Feature Correlation Matrix (zeros filled)")
    fig.tight_layout()
    _save(fig, "correlation.png")


def plot_feature_distributions(df_raw: pd.DataFrame, train_df: pd.DataFrame) -> None:
    """Grid of histograms: raw (with NaN) vs imputed (train split) side-by-side."""
    cols_to_plot = _FEATURE_COLS
    n_cols = 4
    n_rows = int(np.ceil(len(cols_to_plot) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(20, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        ax_raw = axes[i * 2]
        ax_imp = axes[i * 2 + 1]

        raw_vals = df_raw[col].dropna()
        imp_vals = train_df[col].dropna()

        ax_raw.hist(raw_vals, bins=30, color="#4c72b0", alpha=0.8, edgecolor="black")
        ax_raw.set_title(f"{col}\n(raw, NaN dropped)", fontsize=8)
        ax_raw.tick_params(labelsize=7)

        ax_imp.hist(imp_vals, bins=30, color="#dd8452", alpha=0.8, edgecolor="black")
        ax_imp.set_title(f"{col}\n(imputed train)", fontsize=8)
        ax_imp.tick_params(labelsize=7)

    # Hide unused axes
    for j in range(len(cols_to_plot) * 2, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Feature Distributions — Before vs After Imputation", fontsize=12, y=1.01
    )
    fig.tight_layout()
    _save(fig, "feature_distributions.png")


def run_eda() -> None:
    """Entry point: load data, run all plots."""
    logger.info("Starting EDA")
    df_raw = load_pima()
    train_df, _val, _test = impute_and_split(df_raw)

    plot_class_distribution(df_raw)
    plot_correlation(df_raw)
    plot_feature_distributions(df_raw, train_df)
    logger.info("EDA complete", extra={"figures_dir": str(_FIGURES_DIR)})


if __name__ == "__main__":
    run_eda()
