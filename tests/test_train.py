"""Tests for src/training/train.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

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

# Minimal config: n_trials=1 keeps CI fast; lgbm uses tiny tree sizes
_MINIMAL_CONFIG = {
    "optuna": {"n_trials": 1, "direction": "maximize", "metric": "weighted_f1"},
    "model": {
        "lgbm": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "num_leaves": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": -1,
        }
    },
    "mlflow": {
        "experiment_name": "test-experiment",
        "registered_model_name": "TestModel",
    },
}

_BEST_PARAMS = {
    "n_estimators": 10,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


def _make_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    # Alternating labels guarantee both classes in every slice
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


@pytest.fixture()
def tmp_splits(tmp_path, monkeypatch):
    """Write synthetic CSVs to tmp_path and redirect module-level paths."""
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    models = tmp_path / "models"
    models.mkdir()

    df = _make_df(60)
    n = len(df)
    train_df = df.iloc[: int(n * 0.7)].copy()
    val_df = df.iloc[int(n * 0.7) : int(n * 0.85)].copy()
    test_df = df.iloc[int(n * 0.85) :].copy()

    train_df.to_csv(processed / "train.csv", index=False)
    val_df.to_csv(processed / "val.csv", index=False)
    test_df.to_csv(processed / "test.csv", index=False)

    import src.training.train as train_mod

    monkeypatch.setattr(train_mod, "_PROCESSED_DIR", processed)
    monkeypatch.setattr(train_mod, "_MODELS_DIR", models)

    return {
        "tmp": tmp_path,
        "processed": processed,
        "models": models,
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


# ---------- _load_config ----------


def test_load_config_ok(tmp_path):
    from src.training.train import _load_config

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(_MINIMAL_CONFIG))
    cfg = _load_config(str(cfg_file))
    assert "optuna" in cfg
    assert "model" in cfg


def test_load_config_missing_raises(tmp_path):
    from src.exceptions import ConfigError
    from src.training.train import _load_config

    with pytest.raises(ConfigError):
        _load_config(str(tmp_path / "nonexistent.yaml"))


# ---------- _load_splits ----------


def test_load_splits_returns_three_dfs(tmp_splits):
    from src.training.train import _load_splits

    train, val, test = _load_splits()
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


def test_load_splits_missing_raises(tmp_path, monkeypatch):
    import src.training.train as train_mod
    from src.exceptions import ModelNotFoundError
    from src.training.train import _load_splits

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(train_mod, "_PROCESSED_DIR", empty)

    with pytest.raises(ModelNotFoundError):
        _load_splits()


# ---------- train_final_models ----------


def test_train_final_models_saves_joblibs(tmp_splits):
    from src.training.train import train_final_models

    train_final_models(_BEST_PARAMS, tmp_splits["train"], _MINIMAL_CONFIG)

    assert (tmp_splits["models"] / "xgb_model.joblib").exists()
    assert (tmp_splits["models"] / "lgbm_model.joblib").exists()


def test_train_final_models_returns_fitted_models(tmp_splits):
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    from src.training.train import train_final_models

    xgb_m, lgbm_m = train_final_models(
        _BEST_PARAMS, tmp_splits["train"], _MINIMAL_CONFIG
    )
    assert isinstance(xgb_m, XGBClassifier)
    assert isinstance(lgbm_m, LGBMClassifier)


def test_train_final_models_label_assertion(tmp_splits):
    from src.training.train import train_final_models

    bad_df = tmp_splits["train"].copy()
    bad_df["Outcome"] = 2  # invalid label — should trigger assertion
    with pytest.raises(AssertionError):
        train_final_models({}, bad_df, _MINIMAL_CONFIG)


# ---------- run_optuna_study ----------


def test_run_optuna_study_returns_study(tmp_splits):
    import optuna

    import src.training.train as train_mod

    study = train_mod.run_optuna_study(
        tmp_splits["train"], tmp_splits["val"], _MINIMAL_CONFIG
    )
    assert isinstance(study, optuna.Study)
    assert 0.0 <= study.best_value <= 1.0


# ---------- _register_to_mlflow ----------


def test_register_to_mlflow_calls_registry(tmp_splits):
    from xgboost import XGBClassifier

    from src.training.train import _register_to_mlflow

    mock_version = MagicMock()
    mock_version.version = "1"
    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = [mock_version]

    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id"

    with patch("src.training.train.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.start_run.return_value.__exit__.return_value = False
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        study = MagicMock()
        study.best_params = {}
        study.best_value = 0.75

        _register_to_mlflow(XGBClassifier(n_estimators=10), study, _MINIMAL_CONFIG)

        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")
        mock_mlflow.register_model.assert_called_once()
        mock_client.transition_model_version_stage.assert_called_once_with(
            "TestModel", "1", "Staging"
        )
