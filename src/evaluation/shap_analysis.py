"""Day-3 evaluation: SHAP unified API, calibration curve, Brier score."""

import argparse
import json
import pathlib

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt  # isort:skip  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import shap  # noqa: E402
import yaml  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from src.exceptions import ConfigError, ModelNotFoundError  # noqa: E402
from src.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

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
_MODELS_DIR = pathlib.Path("models")
_PROCESSED_DIR = pathlib.Path("data/processed")
_FIGURES_DIR = pathlib.Path("reports/figures")
_RESULTS_PATH = pathlib.Path("reports/results.json")


def _load_config(config_path: str) -> dict:
    path = pathlib.Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config not found: {path}")
    with open(path) as fh:
        return yaml.safe_load(fh)


def _load_artifacts() -> tuple:
    """Load trained models and processed splits from disk."""
    for name in ("xgb_model.joblib", "lgbm_model.joblib", "imputer.joblib"):
        p = _MODELS_DIR / name
        if not p.exists():
            raise ModelNotFoundError(f"Model artifact missing: {p}")
    for split in ("train", "val", "test"):
        p = _PROCESSED_DIR / f"{split}.csv"
        if not p.exists():
            raise ModelNotFoundError(f"Processed split missing: {p}")

    xgb_model = joblib.load(_MODELS_DIR / "xgb_model.joblib")
    lgbm_model = joblib.load(_MODELS_DIR / "lgbm_model.joblib")
    train_df = pd.read_csv(_PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(_PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(_PROCESSED_DIR / "test.csv")
    logger.info(
        "Artifacts loaded",
        extra={"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    )
    return xgb_model, lgbm_model, train_df, val_df, test_df


def compute_shap(model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Compute SHAP values using shap.Explainer unified API.

    SHAP 0.49+ requires a callable rather than the estimator directly;
    predict_proba is used so the output is class probabilities.  The
    full Explanation has shape (n, features, 2); we slice [:, :, 1] to
    keep only class-1 values.  Feature names from the DataFrame columns
    are preserved so dependence plots can index by name.
    """
    feature_names = list(X_train.columns)
    explainer = shap.Explainer(
        model.predict_proba, X_train, feature_names=feature_names
    )
    shap_values_full = explainer(X_test)  # shape: (n, features, 2)
    shap_values = shap_values_full[:, :, 1]  # class-1 slice: (n, features)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(explainer, _MODELS_DIR / "shap_explainer.joblib")
    np.save(str(_MODELS_DIR / "shap_values.npy"), shap_values.values)
    logger.info(
        "SHAP values computed",
        extra={"n_samples": X_test.shape[0], "n_features": X_test.shape[1]},
    )
    return explainer, shap_values


def _plot_beeswarm(shap_values, max_display: int) -> None:
    """Plot 1 — Global beeswarm saved as PNG."""
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.savefig(str(_FIGURES_DIR / "shap_beeswarm.png"), bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Beeswarm plot saved")


def _plot_global_bar(shap_values, feature_names: list) -> dict:
    """Plot 2 — Interactive Plotly bar of mean |SHAP| values (HTML for Streamlit)."""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    feat_imp = dict(zip(feature_names, mean_abs))
    sorted_feat = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
    fig = go.Figure(
        go.Bar(
            x=[v for _, v in sorted_feat],
            y=[k for k, _ in sorted_feat],
            orientation="h",
        )
    )
    fig.update_layout(
        title="Mean |SHAP| Feature Importance",
        xaxis_title="Mean |SHAP value|",
        yaxis_title="Feature",
    )
    fig.write_html(str(_FIGURES_DIR / "shap_global.html"))
    logger.info("Plotly SHAP bar chart saved")
    return feat_imp


def _plot_waterfalls(shap_values, y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """Plot 3 — Waterfall charts for TP, TN, and FP samples."""

    def _first_idx(mask: np.ndarray) -> int:
        idxs = np.where(mask)[0]
        return int(idxs[0]) if len(idxs) > 0 else 0

    tp_idx = _first_idx((y_test == 1) & (y_pred == 1))
    tn_idx = _first_idx((y_test == 0) & (y_pred == 0))
    fp_idx = _first_idx((y_test == 0) & (y_pred == 1))

    for label, idx in [("tp", tp_idx), ("tn", tn_idx), ("fp", fp_idx)]:
        shap.plots.waterfall(shap_values[idx], show=False)
        plt.savefig(
            str(_FIGURES_DIR / f"shap_waterfall_{label}.png"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()
    logger.info(
        "Waterfall plots saved",
        extra={"tp_idx": tp_idx, "tn_idx": tn_idx, "fp_idx": fp_idx},
    )


def _plot_dependence(shap_values, dependence_features: list) -> None:
    """Plot 4 — Dependence scatter plots for each feature in config."""
    for feat in dependence_features:
        shap.plots.scatter(shap_values[:, feat], color=shap_values, show=False)
        plt.savefig(
            str(_FIGURES_DIR / f"shap_dependence_{feat}.png"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()
    logger.info("Dependence plots saved", extra={"features": dependence_features})


def run_calibration(
    xgb_model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
) -> tuple[float, float]:
    """Calibrate XGBoost with isotonic regression; log Brier scores to MLflow."""
    prob_raw = xgb_model.predict_proba(X_test)[:, 1]
    brier_raw = brier_score_loss(y_test, prob_raw)

    cal_model = CalibratedClassifierCV(
        estimator=xgb_model,
        cv="prefit",
        method="isotonic",
    )
    cal_model.fit(X_val, y_val)
    prob_cal = cal_model.predict_proba(X_test)[:, 1]
    brier_cal = brier_score_loss(y_test, prob_cal)

    fig, ax = plt.subplots()
    CalibrationDisplay.from_predictions(
        y_test, prob_raw, n_bins=10, ax=ax, label="XGB raw"
    )
    CalibrationDisplay.from_predictions(
        y_test, prob_cal, n_bins=10, ax=ax, label="XGB calibrated"
    )
    ax.set_title("Calibration Curve")
    plt.savefig(
        str(_FIGURES_DIR / "calibration_curve.png"), bbox_inches="tight", dpi=150
    )
    plt.close()

    joblib.dump(cal_model, _MODELS_DIR / "calibrated_xgb.joblib")

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metric("brier_score_raw", brier_raw)
        mlflow.log_metric("brier_score_calibrated", brier_cal)

    logger.info(
        "Calibration complete",
        extra={
            "brier_raw": round(brier_raw, 4),
            "brier_cal": round(brier_cal, 4),
        },
    )
    return brier_raw, brier_cal


def _update_results_json(
    xgb_model,
    lgbm_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    brier_xgb: float,
    feat_imp: dict,
) -> None:
    """Write reports/results.json with per-model metrics and top SHAP features."""
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    lgbm_pred = lgbm_model.predict(X_test)
    lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
    lgbm_brier = brier_score_loss(y_test, lgbm_proba)

    top5 = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:5]
    results = {
        "xgboost": {
            "accuracy": float(accuracy_score(y_test, xgb_pred)),
            "weighted_f1": float(f1_score(y_test, xgb_pred, average="weighted")),
            "auc_roc": float(roc_auc_score(y_test, xgb_proba)),
            "brier_score": float(brier_xgb),
        },
        "lightgbm": {
            "accuracy": float(accuracy_score(y_test, lgbm_pred)),
            "weighted_f1": float(f1_score(y_test, lgbm_pred, average="weighted")),
            "auc_roc": float(roc_auc_score(y_test, lgbm_proba)),
            "brier_score": float(lgbm_brier),
        },
        "top_shap_features": [k for k, _ in top5],
    }
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("results.json updated", extra={"path": str(_RESULTS_PATH)})


def main(config_path: str = "config/config.yaml") -> None:
    """Day-3 evaluation pipeline: SHAP + calibration + metrics."""
    config = _load_config(config_path)
    xgb_model, lgbm_model, train_df, val_df, test_df = _load_artifacts()

    # DataFrames preserve feature names for SHAP dependence slice-by-name
    X_train_df = train_df[_FEATURE_COLS]
    X_test_df = test_df[_FEATURE_COLS]
    X_val = val_df[_FEATURE_COLS].values
    y_val = val_df["Outcome"].values
    X_test = X_test_df.values
    y_test = test_df["Outcome"].values

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    _, shap_values = compute_shap(xgb_model, X_train_df, X_test_df)

    shap_cfg = config["shap"]
    _plot_beeswarm(shap_values, shap_cfg["max_display"])
    feat_imp = _plot_global_bar(shap_values, _FEATURE_COLS)

    y_pred = xgb_model.predict(X_test)
    _plot_waterfalls(shap_values, y_test, y_pred)
    _plot_dependence(shap_values, shap_cfg["dependence_features"])

    sorted_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
    with open(_FIGURES_DIR / "feature_importance_shap.json", "w") as f:
        json.dump(sorted_imp, f, indent=2)

    brier_raw, _ = run_calibration(xgb_model, X_val, y_val, X_test, y_test, config)
    _update_results_json(xgb_model, lgbm_model, X_test, y_test, brier_raw, feat_imp)
    logger.info("Day 3 evaluation pipeline complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day-3 SHAP evaluation pipeline")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config/config.yaml",
    )
    args = parser.parse_args()
    main(args.config)
