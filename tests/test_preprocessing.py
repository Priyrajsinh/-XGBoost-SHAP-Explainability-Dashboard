"""Tests for src/data/preprocessing.py — file I/O is redirected to tmp_path."""

import numpy as np
import pandas as pd
import pytest

_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

# 40 synthetic rows so stratified splits work (need >=2 per class in each split)
_RNG = np.random.default_rng(0)


def _make_df(n: int = 80) -> pd.DataFrame:
    data = {
        "Pregnancies": _RNG.integers(0, 10, n).astype(int),
        "Glucose": _RNG.uniform(80, 200, n),
        "BloodPressure": _RNG.uniform(50, 100, n),
        "SkinThickness": _RNG.uniform(10, 50, n),
        "Insulin": _RNG.uniform(0, 200, n),
        "BMI": _RNG.uniform(18, 45, n),
        "DiabetesPedigreeFunction": _RNG.uniform(0.1, 2.5, n),
        "Age": _RNG.integers(21, 70, n).astype(int),
        "Outcome": ([0] * (n // 2) + [1] * (n // 2)),
    }
    df = pd.DataFrame(data)
    # Introduce some NaN to simulate zero-as-nan replacement
    df.loc[df.index[:5], "Insulin"] = np.nan
    df.loc[df.index[:3], "Glucose"] = np.nan
    return df


@pytest.fixture()
def tmp_dirs(tmp_path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    models = tmp_path / "models"
    models.mkdir()
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "config.yaml"
    cfg_file.write_text(
        "data:\n"
        "  zero_as_nan_cols: [Glucose, BloodPressure, SkinThickness, Insulin, BMI]\n"
        "  seed: 42\n"
        "  train_split: 0.70\n"
        "  val_split: 0.15\n"
        "  test_split: 0.15\n"
    )
    import src.data.preprocessing as pp_mod

    monkeypatch.setattr(pp_mod, "_PROCESSED_DIR", processed)
    monkeypatch.setattr(pp_mod, "_MODELS_DIR", models)
    monkeypatch.setattr(pp_mod, "_IMPUTER_PATH", models / "imputer.joblib")
    monkeypatch.setattr(pp_mod, "_CONFIG_PATH", cfg_file)
    return tmp_path


def test_impute_and_split_returns_three_dfs(tmp_dirs):
    from src.data.preprocessing import impute_and_split

    df = _make_df()
    train, val, test = impute_and_split(df)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


def test_split_sizes_sum_to_total(tmp_dirs):
    from src.data.preprocessing import impute_and_split

    df = _make_df()
    train, val, test = impute_and_split(df)
    assert len(train) + len(val) + len(test) == len(df)


def test_no_nan_after_imputation(tmp_dirs):
    from src.data.preprocessing import _FEATURE_COLS, impute_and_split

    df = _make_df()
    train, val, test = impute_and_split(df)
    for split in (train, val, test):
        assert split[_FEATURE_COLS].isna().sum().sum() == 0


def test_no_overlap_between_splits(tmp_dirs):
    from src.data.preprocessing import impute_and_split

    df = _make_df()
    train, val, test = impute_and_split(df)
    t_idx = set(train.index)
    v_idx = set(val.index)
    te_idx = set(test.index)
    assert len(t_idx & v_idx) == 0
    assert len(t_idx & te_idx) == 0
    assert len(v_idx & te_idx) == 0


def test_processed_csvs_saved(tmp_dirs):
    import src.data.preprocessing as pp_mod
    from src.data.preprocessing import impute_and_split

    df = _make_df()
    impute_and_split(df)
    assert (pp_mod._PROCESSED_DIR / "train.csv").exists()
    assert (pp_mod._PROCESSED_DIR / "val.csv").exists()
    assert (pp_mod._PROCESSED_DIR / "test.csv").exists()


def test_imputer_joblib_saved(tmp_dirs):
    import src.data.preprocessing as pp_mod
    from src.data.preprocessing import impute_and_split

    df = _make_df()
    impute_and_split(df)
    assert pp_mod._IMPUTER_PATH.exists()


def test_outcome_col_intact_after_imputation(tmp_dirs):
    from src.data.preprocessing import impute_and_split

    df = _make_df()
    train, val, test = impute_and_split(df)
    for split in (train, val, test):
        assert "Outcome" in split.columns
        assert set(split["Outcome"].unique()).issubset({0, 1})


def test_stratified_class_balance(tmp_dirs):
    """Val and test should both contain both classes (stratified)."""
    from src.data.preprocessing import impute_and_split

    df = _make_df(n=100)
    train, val, test = impute_and_split(df)
    assert set(val["Outcome"].unique()) == {0, 1}
    assert set(test["Outcome"].unique()) == {0, 1}
