"""Tests for src/evaluation/evaluate.py."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_test_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    outcomes = ([0, 1] * (n // 2))[:n]
    return pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 10, n).astype(int),
            "Glucose": rng.uniform(80, 200, n),
            "BloodPressure": rng.uniform(50, 100, n),
            "SkinThickness": rng.uniform(10, 50, n),
            "Insulin": rng.uniform(0, 200, n),
            "BMI": rng.uniform(18, 45, n),
            "DiabetesPedigreeFunction": rng.uniform(0.1, 2.5, n),
            "Age": rng.integers(21, 70, n).astype(int),
            "Outcome": outcomes,
        }
    )


def _make_mock_model(n_samples: int = 40):
    """Return a MagicMock that behaves like a fitted sklearn classifier."""
    rng = np.random.default_rng(0)
    mock = MagicMock()
    preds = rng.integers(0, 2, n_samples)
    proba_1 = rng.uniform(0.1, 0.9, n_samples)
    proba_0 = 1 - proba_1
    mock.predict.return_value = preds
    mock.predict_proba.return_value = np.column_stack([proba_0, proba_1])
    return mock


@pytest.fixture()
def eval_env(tmp_path, monkeypatch):
    """Write synthetic CSVs and redirect module-level paths in evaluate.py."""
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    figures_dir = tmp_path / "reports" / "figures"
    figures_dir.mkdir(parents=True)
    reports_dir = tmp_path / "reports"

    df = _make_test_df(40)
    df.to_csv(processed / "test.csv", index=False)

    import src.evaluation.evaluate as ev

    monkeypatch.setattr(ev, "_PROCESSED_DIR", processed)
    monkeypatch.setattr(ev, "_MODELS_DIR", models_dir)
    monkeypatch.setattr(ev, "_REPORTS_DIR", reports_dir)
    monkeypatch.setattr(ev, "_FIGURES_DIR", figures_dir)
    monkeypatch.setattr(ev, "_RESULTS_FILE", reports_dir / "results.json")

    return {
        "tmp": tmp_path,
        "processed": processed,
        "models_dir": models_dir,
        "figures_dir": figures_dir,
        "reports_dir": reports_dir,
        "df": df,
    }


# ---------------------------------------------------------------------------
# _load_test_data
# ---------------------------------------------------------------------------


def test_load_test_data_returns_arrays(eval_env):
    from src.evaluation.evaluate import _load_test_data

    X, y = _load_test_data()
    assert X.shape == (40, 8)
    assert y.shape == (40,)


def test_load_test_data_missing_raises(tmp_path, monkeypatch):
    import src.evaluation.evaluate as ev
    from src.evaluation.evaluate import _load_test_data
    from src.exceptions import ModelNotFoundError

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(ev, "_PROCESSED_DIR", empty)

    with pytest.raises(ModelNotFoundError):
        _load_test_data()


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_keys(eval_env):
    from src.evaluation.evaluate import _compute_metrics, _load_test_data

    X, y = _load_test_data()
    model = _make_mock_model(len(y))
    # fix predict to return matching-length array
    model.predict.return_value = np.zeros(len(y), dtype=int)
    proba_0 = np.full(len(y), 0.4)
    proba_1 = np.full(len(y), 0.6)
    model.predict_proba.return_value = np.column_stack([proba_0, proba_1])

    metrics = _compute_metrics(model, X, y)
    assert set(metrics.keys()) == {"accuracy", "weighted_f1", "auc_roc", "brier_score"}


def test_compute_metrics_values_in_range(eval_env):
    from src.evaluation.evaluate import _compute_metrics, _load_test_data

    X, y = _load_test_data()
    model = _make_mock_model(len(y))
    metrics = _compute_metrics(model, X, y)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["weighted_f1"] <= 1.0
    assert 0.0 <= metrics["auc_roc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0


# ---------------------------------------------------------------------------
# _save_confusion_matrix
# ---------------------------------------------------------------------------


def test_save_confusion_matrix_creates_png(eval_env):
    from src.evaluation.evaluate import (
        _load_test_data,
        _save_confusion_matrix,
    )

    X, y = _load_test_data()
    model = _make_mock_model(len(y))
    model.predict.return_value = np.zeros(len(y), dtype=int)

    out = _save_confusion_matrix(model, X, y, "xgb_model")
    assert out.exists()
    assert out.suffix == ".png"


# ---------------------------------------------------------------------------
# _save_roc_curves
# ---------------------------------------------------------------------------


def test_save_roc_curves_creates_png(eval_env):
    from src.evaluation.evaluate import _load_test_data, _save_roc_curves

    X, y = _load_test_data()
    n = len(y)
    models = {
        "xgb_model": _make_mock_model(n),
        "lgbm_model": _make_mock_model(n),
        "calibrated_xgb": _make_mock_model(n),
    }

    out = _save_roc_curves(models, X, y)
    assert out.exists()
    assert out.name == "roc_curves.png"


# ---------------------------------------------------------------------------
# main (mocked MLflow)
# ---------------------------------------------------------------------------


def test_main_updates_results_json(eval_env, monkeypatch):
    from src.evaluation.evaluate import main

    n = 40
    mock_model = _make_mock_model(n)

    def fake_load(path):
        return mock_model

    import src.evaluation.evaluate as ev

    monkeypatch.setattr(
        ev,
        "_load_models",
        lambda: {
            "xgb_model": _make_mock_model(n),
            "lgbm_model": _make_mock_model(n),
            "calibrated_xgb": _make_mock_model(n),
        },
    )

    with patch("src.evaluation.evaluate.mlflow"):
        main()

    results_path = eval_env["reports_dir"] / "results.json"
    assert results_path.exists()
    data = json.loads(results_path.read_text())
    assert "xgboost" in data
    assert "lightgbm" in data
    assert "calibrated_xgb" in data
