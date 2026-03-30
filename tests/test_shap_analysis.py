"""Tests for src/evaluation/shap_analysis.py."""

import json
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

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

_MINIMAL_CONFIG = {
    "shap": {"max_display": 5, "dependence_features": ["Glucose"]},
    "mlflow": {"experiment_name": "test-exp"},
}


def _make_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    outcomes = ([0, 1] * (n // 2))[:n]
    return pd.DataFrame(
        {
            "Pregnancies": rng.uniform(0, 10, n),
            "Glucose": rng.uniform(80, 200, n),
            "BloodPressure": rng.uniform(50, 100, n),
            "SkinThickness": rng.uniform(10, 50, n),
            "Insulin": rng.uniform(0, 200, n),
            "BMI": rng.uniform(18, 45, n),
            "DiabetesPedigreeFunction": rng.uniform(0.1, 2.5, n),
            "Age": rng.uniform(21, 70, n),
            "Outcome": outcomes,
        }
    )


@pytest.fixture()
def eval_env(tmp_path, monkeypatch):
    """Set up dirs, tiny models, processed splits and patch module-level paths."""
    import src.evaluation.shap_analysis as mod

    models_dir = tmp_path / "models"
    figures_dir = tmp_path / "reports" / "figures"
    processed_dir = tmp_path / "data" / "processed"
    results_path = tmp_path / "reports" / "results.json"
    for d in (models_dir, figures_dir, processed_dir, results_path.parent):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "_MODELS_DIR", models_dir)
    monkeypatch.setattr(mod, "_PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(mod, "_FIGURES_DIR", figures_dir)
    monkeypatch.setattr(mod, "_RESULTS_PATH", results_path)

    df = _make_df(80)
    n = len(df)
    train_df = df.iloc[: int(n * 0.7)].reset_index(drop=True)
    val_df = df.iloc[int(n * 0.7) : int(n * 0.85)].reset_index(drop=True)
    test_df = df.iloc[int(n * 0.85) :].reset_index(drop=True)

    X_train = train_df[_FEATURE_COLS].values
    y_train = train_df["Outcome"].values

    xgb_model = XGBClassifier(
        n_estimators=5, max_depth=2, random_state=42, eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)

    lgbm_model = LGBMClassifier(
        n_estimators=5, max_depth=2, random_state=42, verbosity=-1
    )
    lgbm_model.fit(X_train, y_train)

    joblib.dump(xgb_model, models_dir / "xgb_model.joblib")
    joblib.dump(lgbm_model, models_dir / "lgbm_model.joblib")
    (models_dir / "imputer.joblib").touch()  # existence-only check in _load_artifacts

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    return {
        "xgb": xgb_model,
        "lgbm": lgbm_model,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "models_dir": models_dir,
        "figures_dir": figures_dir,
        "processed_dir": processed_dir,
        "results_path": results_path,
    }


@pytest.fixture()
def shap_vals(eval_env):
    """Pre-computed SHAP values for reuse in plot-level tests."""
    from src.evaluation.shap_analysis import compute_shap

    env = eval_env
    X_train = env["train_df"][_FEATURE_COLS]
    X_test = env["test_df"][_FEATURE_COLS]
    _, sv = compute_shap(env["xgb"], X_train, X_test)
    return sv, env


# ---------- _load_config ----------


def test_load_config_ok(tmp_path):
    from src.evaluation.shap_analysis import _load_config

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(_MINIMAL_CONFIG))
    cfg = _load_config(str(cfg_file))
    assert "shap" in cfg
    assert "mlflow" in cfg


def test_load_config_missing_raises(tmp_path):
    from src.evaluation.shap_analysis import _load_config
    from src.exceptions import ConfigError

    with pytest.raises(ConfigError):
        _load_config(str(tmp_path / "nonexistent.yaml"))


# ---------- _load_artifacts ----------


def test_load_artifacts_missing_model_raises(tmp_path, monkeypatch):
    import src.evaluation.shap_analysis as mod
    from src.evaluation.shap_analysis import _load_artifacts
    from src.exceptions import ModelNotFoundError

    empty_models = tmp_path / "models"
    empty_models.mkdir()
    monkeypatch.setattr(mod, "_MODELS_DIR", empty_models)

    with pytest.raises(ModelNotFoundError):
        _load_artifacts()


def test_load_artifacts_missing_split_raises(tmp_path, monkeypatch):
    import src.evaluation.shap_analysis as mod
    from src.evaluation.shap_analysis import _load_artifacts
    from src.exceptions import ModelNotFoundError

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for name in ("xgb_model.joblib", "lgbm_model.joblib", "imputer.joblib"):
        (models_dir / name).touch()
    empty_processed = tmp_path / "processed"
    empty_processed.mkdir()

    monkeypatch.setattr(mod, "_MODELS_DIR", models_dir)
    monkeypatch.setattr(mod, "_PROCESSED_DIR", empty_processed)

    with pytest.raises(ModelNotFoundError):
        _load_artifacts()


# ---------- compute_shap ----------


def test_compute_shap_saves_files(eval_env):
    from src.evaluation.shap_analysis import compute_shap

    env = eval_env
    X_train = env["train_df"][_FEATURE_COLS]
    X_test = env["test_df"][_FEATURE_COLS]
    compute_shap(env["xgb"], X_train, X_test)

    assert (env["models_dir"] / "shap_explainer.joblib").exists()
    assert (env["models_dir"] / "shap_values.npy").exists()


def test_compute_shap_values_shape(eval_env):
    from src.evaluation.shap_analysis import compute_shap

    env = eval_env
    X_train = env["train_df"][_FEATURE_COLS]
    X_test = env["test_df"][_FEATURE_COLS]
    _, sv = compute_shap(env["xgb"], X_train, X_test)

    assert sv.values.shape == (len(X_test), len(_FEATURE_COLS))


# ---------- plot functions ----------


def test_plot_beeswarm_creates_png(shap_vals):
    from src.evaluation.shap_analysis import _plot_beeswarm

    sv, env = shap_vals
    _plot_beeswarm(sv, max_display=5)
    assert (env["figures_dir"] / "shap_beeswarm.png").exists()


def test_plot_global_bar_creates_html_and_returns_dict(shap_vals):
    from src.evaluation.shap_analysis import _plot_global_bar

    sv, env = shap_vals
    feat_imp = _plot_global_bar(sv, _FEATURE_COLS)

    assert (env["figures_dir"] / "shap_global.html").exists()
    assert isinstance(feat_imp, dict)
    assert set(feat_imp.keys()) == set(_FEATURE_COLS)


def test_plot_waterfalls_creates_three_pngs(shap_vals):
    from src.evaluation.shap_analysis import _plot_waterfalls

    sv, env = shap_vals
    y_test = env["test_df"]["Outcome"].values
    y_pred = env["xgb"].predict(env["test_df"][_FEATURE_COLS].values)
    _plot_waterfalls(sv, y_test, y_pred)

    for label in ("tp", "tn", "fp"):
        assert (env["figures_dir"] / f"shap_waterfall_{label}.png").exists()


def test_plot_dependence_creates_png(shap_vals):
    from src.evaluation.shap_analysis import _plot_dependence

    sv, env = shap_vals
    _plot_dependence(sv, ["Glucose"])
    assert (env["figures_dir"] / "shap_dependence_Glucose.png").exists()


# ---------- run_calibration ----------


def test_run_calibration_returns_float_pair(eval_env):
    from src.evaluation.shap_analysis import run_calibration

    env = eval_env
    X_val = env["val_df"][_FEATURE_COLS].values
    y_val = env["val_df"]["Outcome"].values
    X_test = env["test_df"][_FEATURE_COLS].values
    y_test = env["test_df"]["Outcome"].values

    with patch("src.evaluation.shap_analysis.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
        mock_mlflow.start_run.return_value.__exit__.return_value = False
        brier_raw, brier_cal = run_calibration(
            env["xgb"], X_val, y_val, X_test, y_test, _MINIMAL_CONFIG
        )

    assert isinstance(brier_raw, float)
    assert isinstance(brier_cal, float)
    assert 0.0 <= brier_raw <= 1.0
    assert 0.0 <= brier_cal <= 1.0


def test_run_calibration_saves_artifacts(eval_env):
    from src.evaluation.shap_analysis import run_calibration

    env = eval_env
    X_val = env["val_df"][_FEATURE_COLS].values
    y_val = env["val_df"]["Outcome"].values
    X_test = env["test_df"][_FEATURE_COLS].values
    y_test = env["test_df"]["Outcome"].values

    with patch("src.evaluation.shap_analysis.mlflow"):
        run_calibration(env["xgb"], X_val, y_val, X_test, y_test, _MINIMAL_CONFIG)

    assert (env["models_dir"] / "calibrated_xgb.joblib").exists()
    assert (env["figures_dir"] / "calibration_curve.png").exists()


# ---------- _update_results_json ----------


def test_update_results_json_has_required_keys(eval_env):
    from src.evaluation.shap_analysis import _update_results_json

    env = eval_env
    X_test = env["test_df"][_FEATURE_COLS].values
    y_test = env["test_df"]["Outcome"].values
    feat_imp = {col: float(i) for i, col in enumerate(_FEATURE_COLS)}

    _update_results_json(env["xgb"], env["lgbm"], X_test, y_test, 0.2, feat_imp)

    assert env["results_path"].exists()
    data = json.loads(env["results_path"].read_text())
    assert "xgboost" in data
    assert "lightgbm" in data
    assert "top_shap_features" in data
    for model_key in ("xgboost", "lightgbm"):
        for metric in ("accuracy", "weighted_f1", "auc_roc", "brier_score"):
            assert metric in data[model_key]


def test_update_results_json_top5_features(eval_env):
    from src.evaluation.shap_analysis import _update_results_json

    env = eval_env
    X_test = env["test_df"][_FEATURE_COLS].values
    y_test = env["test_df"]["Outcome"].values
    feat_imp = {col: float(i) for i, col in enumerate(_FEATURE_COLS)}

    _update_results_json(env["xgb"], env["lgbm"], X_test, y_test, 0.2, feat_imp)
    data = json.loads(env["results_path"].read_text())
    assert len(data["top_shap_features"]) == 5
