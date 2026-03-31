"""Smoke tests for XGBoost, MeanMedianImputer, and SHAP values — self-contained."""

import pathlib

import joblib
import numpy as np
import pandas as pd
import shap
from feature_engine.imputation import MeanMedianImputer
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


def _make_df(n: int = 40) -> pd.DataFrame:
    """Return a synthetic DataFrame with 8 Pima features and balanced labels."""
    rng = np.random.default_rng(0)
    outcomes = ([0, 1] * (n // 2))[:n]
    return pd.DataFrame(
        {col: rng.uniform(50, 150, n) for col in _FEATURE_COLS} | {"Outcome": outcomes}
    )


def test_xgb_model_loads(tmp_path: pathlib.Path) -> None:
    """XGBoost model can be serialised and deserialised without error."""
    df = _make_df()
    X = df[_FEATURE_COLS].values
    y = df["Outcome"].values

    model = XGBClassifier(
        n_estimators=5, max_depth=2, random_state=42, eval_metric="logloss"
    )
    model.fit(X, y)

    out = tmp_path / "xgb_model.joblib"
    joblib.dump(model, out)
    loaded = joblib.load(out)

    assert loaded is not None
    preds = loaded.predict(X)
    assert len(preds) == len(y)


def test_imputer_no_nan(tmp_path: pathlib.Path) -> None:
    """Fitted MeanMedianImputer transforms data with NaN to produce no NaN."""
    df = _make_df()
    # introduce NaN in a few cells (simulating zero-as-NaN replacement)
    df.loc[0, "Glucose"] = np.nan
    df.loc[2, "Insulin"] = np.nan

    imputer = MeanMedianImputer(
        imputation_method="median",
        variables=_FEATURE_COLS,  # type: ignore[arg-type]
    )
    imputer.fit(df[_FEATURE_COLS])

    result = imputer.transform(df[_FEATURE_COLS])
    assert not np.any(np.isnan(result.values)), "Imputer left NaN in output"


def test_shap_values_shape() -> None:
    """SHAP values computed from a fitted XGBoost model have shape (n_samples, 8)."""
    df = _make_df()
    X = df[_FEATURE_COLS]
    y = df["Outcome"].values

    model = XGBClassifier(
        n_estimators=5, max_depth=2, random_state=42, eval_metric="logloss"
    )
    model.fit(X, y)

    explainer = shap.Explainer(model.predict_proba, X, feature_names=_FEATURE_COLS)
    sv = explainer(X)

    # sv.values shape: (n_samples, n_features) or (n_samples, n_features, n_classes)
    assert sv.values.shape[1] == len(
        _FEATURE_COLS
    ), f"Expected {len(_FEATURE_COLS)} feature cols, got {sv.values.shape[1]}"
