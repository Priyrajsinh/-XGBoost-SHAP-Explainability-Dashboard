"""Tests for src/data/dataset.py — network and file I/O are mocked."""

import json
from unittest.mock import patch

import pandas as pd
import pytest

_SAMPLE_ROWS = [
    [6, 148, 72, 35, 0, 33.6, 0.627, 50, 1],
    [1, 85, 66, 29, 0, 26.6, 0.351, 31, 0],
    [8, 183, 64, 0, 0, 23.3, 0.672, 32, 1],
    [1, 89, 66, 23, 94, 28.1, 0.167, 21, 0],
    [0, 137, 40, 35, 168, 43.1, 2.288, 33, 1],
]
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


def _make_csv_bytes() -> bytes:
    lines = [",".join(map(str, row)) for row in _SAMPLE_ROWS]
    return "\n".join(lines).encode()


@pytest.fixture()
def tmp_dirs(tmp_path, monkeypatch):
    """Redirect data/raw and config paths to tmp_path."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
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
    import src.data.dataset as ds_mod

    monkeypatch.setattr(ds_mod, "_RAW_DIR", raw_dir)
    monkeypatch.setattr(ds_mod, "_PIMA_CSV", raw_dir / "pima.csv")
    monkeypatch.setattr(ds_mod, "_CHECKSUMS_JSON", raw_dir / "checksums.json")
    monkeypatch.setattr(ds_mod, "_CONFIG_PATH", cfg_file)
    return tmp_path


def test_load_pima_returns_dataframe(tmp_dirs):
    with patch(
        "pandas.read_csv", return_value=pd.DataFrame(_SAMPLE_ROWS, columns=_COLS)
    ):
        from src.data.dataset import load_pima

        df = load_pima()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == _COLS


def test_load_pima_zeros_replaced_with_nan(tmp_dirs):
    with patch(
        "pandas.read_csv", return_value=pd.DataFrame(_SAMPLE_ROWS, columns=_COLS)
    ):
        from src.data.dataset import load_pima

        df = load_pima()
    # Rows 0,1,2 had 0 Insulin — should be NaN after processing
    assert df["Insulin"].isna().sum() >= 1


def test_load_pima_checksum_saved(tmp_dirs):
    import src.data.dataset as ds_mod

    with patch(
        "pandas.read_csv", return_value=pd.DataFrame(_SAMPLE_ROWS, columns=_COLS)
    ):
        from src.data.dataset import load_pima

        load_pima()
    ck_path = ds_mod._CHECKSUMS_JSON
    assert ck_path.exists()
    with open(ck_path) as fh:
        data = json.load(fh)
    assert "pima.csv" in data
    assert len(data["pima.csv"]) == 64  # SHA-256 hex length


def test_load_pima_csv_saved(tmp_dirs):
    import src.data.dataset as ds_mod

    with patch(
        "pandas.read_csv", return_value=pd.DataFrame(_SAMPLE_ROWS, columns=_COLS)
    ):
        from src.data.dataset import load_pima

        load_pima()
    assert ds_mod._PIMA_CSV.exists()


def test_load_pima_raises_on_fetch_error(tmp_dirs):
    with patch("pandas.read_csv", side_effect=Exception("network error")):
        from src.data.dataset import load_pima
        from src.exceptions import DataLoadError

        with pytest.raises(DataLoadError):
            load_pima()


def test_load_pima_outcome_col_present(tmp_dirs):
    with patch(
        "pandas.read_csv", return_value=pd.DataFrame(_SAMPLE_ROWS, columns=_COLS)
    ):
        from src.data.dataset import load_pima

        df = load_pima()
    assert "Outcome" in df.columns
    assert set(df["Outcome"].unique()).issubset({0, 1})
