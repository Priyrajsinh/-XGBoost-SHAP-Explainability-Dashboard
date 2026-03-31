"""Smoke tests for serialised model artifacts in models/."""

import pathlib

import joblib
import numpy as np

_MODELS_DIR = pathlib.Path("models")
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


def test_xgb_model_loads() -> None:
    """XGBoost model artifact can be deserialised without error."""
    model = joblib.load(_MODELS_DIR / "xgb_model.joblib")
    assert model is not None


def test_imputer_no_nan() -> None:
    """Fitted imputer transforms a row with NaN values to produce no NaN."""
    imputer = joblib.load(_MODELS_DIR / "imputer.joblib")
    rng = np.random.default_rng(0)
    X_sample = rng.uniform(50, 150, size=(5, len(_FEATURE_COLS)))
    # introduce NaN in a few cells
    X_sample[0, 1] = np.nan
    X_sample[2, 4] = np.nan

    import pandas as pd

    df = pd.DataFrame(X_sample, columns=_FEATURE_COLS)
    result = imputer.transform(df)
    assert not np.any(np.isnan(result.values)), "Imputer left NaN in output"


def test_shap_values_shape() -> None:
    """Saved SHAP values array has exactly 8 feature columns."""
    shap_vals = np.load(_MODELS_DIR / "shap_values.npy")
    assert shap_vals.shape[1] == len(
        _FEATURE_COLS
    ), f"Expected {len(_FEATURE_COLS)} feature cols, got {shap_vals.shape[1]}"
