"""Tests for src/data/eda.py — all I/O and rendering mocked."""

import numpy as np
import pandas as pd
import pytest


def _make_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 10, n).astype(int),
            "Glucose": rng.uniform(80, 200, n),
            "BloodPressure": rng.uniform(50, 100, n),
            "SkinThickness": rng.uniform(10, 50, n),
            "Insulin": rng.uniform(20, 200, n),
            "BMI": rng.uniform(18, 45, n),
            "DiabetesPedigreeFunction": rng.uniform(0.1, 2.5, n),
            "Age": rng.integers(21, 70, n).astype(int),
            "Outcome": ([0] * (n // 2) + [1] * (n // 2)),
        }
    )


@pytest.fixture()
def patch_figures_dir(tmp_path, monkeypatch):
    import src.data.eda as eda_mod

    monkeypatch.setattr(eda_mod, "_FIGURES_DIR", tmp_path / "figures")
    return tmp_path


def test_plot_class_distribution_saves_file(patch_figures_dir, tmp_path):
    from src.data.eda import plot_class_distribution

    df = _make_df()
    plot_class_distribution(df)
    assert (tmp_path / "figures" / "class_dist.png").exists()


def test_plot_correlation_saves_file(patch_figures_dir, tmp_path):
    from src.data.eda import plot_correlation

    df = _make_df()
    plot_correlation(df)
    assert (tmp_path / "figures" / "correlation.png").exists()


def test_plot_feature_distributions_saves_file(patch_figures_dir, tmp_path):
    from src.data.eda import plot_feature_distributions

    df = _make_df()
    plot_feature_distributions(df, df)
    assert (tmp_path / "figures" / "feature_distributions.png").exists()


def test_run_eda_calls_all_plots(patch_figures_dir, monkeypatch):
    """run_eda should produce all three figures given mocked data sources."""
    import src.data.eda as eda_mod

    df = _make_df()
    train_df = df.iloc[:35].copy()

    monkeypatch.setattr(eda_mod, "load_pima", lambda: df)
    monkeypatch.setattr(
        eda_mod, "impute_and_split", lambda x: (train_df, df.iloc[35:42], df.iloc[42:])
    )

    eda_mod.run_eda()

    figures_dir = patch_figures_dir / "figures"
    assert (figures_dir / "class_dist.png").exists()
    assert (figures_dir / "correlation.png").exists()
    assert (figures_dir / "feature_distributions.png").exists()
