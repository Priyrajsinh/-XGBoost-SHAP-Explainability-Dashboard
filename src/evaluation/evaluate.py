"""Day 5 — 3-model evaluation: XGBoost, LightGBM, calibrated_xgb."""

import json
import pathlib

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt  # isort:skip  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
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
_REPORTS_DIR = pathlib.Path("reports")
_FIGURES_DIR = _REPORTS_DIR / "figures"
_RESULTS_FILE = _REPORTS_DIR / "results.json"

# Human-readable names for confusion-matrix filenames and plot titles
_DISPLAY_NAMES = {
    "xgb_model": "XGBoost",
    "lgbm_model": "LightGBM",
    "calibrated_xgb": "XGBoost (calibrated)",
}
# Keys used in results.json
_RESULT_KEYS = {
    "xgb_model": "xgboost",
    "lgbm_model": "lightgbm",
    "calibrated_xgb": "calibrated_xgb",
}


def _load_config() -> dict:
    p = pathlib.Path("config/config.yaml")
    if not p.exists():
        raise ConfigError(f"Config not found: {p}")
    with open(p) as fh:
        return yaml.safe_load(fh)


def _load_test_data() -> tuple:
    """Return (X_test ndarray, y_test ndarray) from data/processed/test.csv."""
    p = _PROCESSED_DIR / "test.csv"
    if not p.exists():
        raise ModelNotFoundError(f"Test split missing: {p}")
    df = pd.read_csv(p)
    X_test = df[_FEATURE_COLS].values
    y_test = df["Outcome"].values
    logger.info("test_data_loaded", extra={"rows": len(df)})
    return X_test, y_test


def _load_models() -> dict:
    """Load xgb_model, lgbm_model, calibrated_xgb from models/."""
    names = ("xgb_model", "lgbm_model", "calibrated_xgb")
    models = {}
    for name in names:
        p = _MODELS_DIR / f"{name}.joblib"
        if not p.exists():
            raise ModelNotFoundError(f"Model file missing: {p}")
        models[name] = joblib.load(p)
        logger.info("model_loaded", extra={"model": name})
    return models


def _compute_metrics(model, X_test, y_test) -> dict:
    """Return accuracy, weighted_f1, auc_roc, brier_score for one model."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "weighted_f1": round(float(f1_score(y_test, preds, average="weighted")), 4),
        "auc_roc": round(float(roc_auc_score(y_test, proba)), 4),
        "brier_score": round(float(brier_score_loss(y_test, proba)), 4),
    }


def _save_confusion_matrix(model, X_test, y_test, model_key: str) -> pathlib.Path:
    """Save confusion-matrix PNG for one model; return output path."""
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Diabetic", "Diabetic"],
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {_DISPLAY_NAMES[model_key]}")
    fig.tight_layout()
    out = _FIGURES_DIR / f"confusion_matrix_{model_key}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("confusion_matrix_saved", extra={"path": str(out)})
    return out


def _save_roc_curves(models: dict, X_test, y_test) -> pathlib.Path:
    """Plot all 3 ROC curves on one axes; return output path."""
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    for key, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{_DISPLAY_NAMES[key]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — 3 Model Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = _FIGURES_DIR / "roc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("roc_curves_saved", extra={"path": str(out)})
    return out


def _log_to_mlflow(all_metrics: dict, config: dict) -> None:
    """Log every model's metrics + figure artifacts to one MLflow run."""
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="evaluation_3model"):
        for model_key, metrics in all_metrics.items():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{model_key}_{metric_name}", value)
        roc_path = _FIGURES_DIR / "roc_curves.png"
        if roc_path.exists():
            mlflow.log_artifact(str(roc_path))
        for name in ("xgb_model", "lgbm_model", "calibrated_xgb"):
            cm_path = _FIGURES_DIR / f"confusion_matrix_{name}.png"
            if cm_path.exists():
                mlflow.log_artifact(str(cm_path))
    logger.info("mlflow_evaluation_logged")


def main() -> None:
    """Run 3-model evaluation, update results.json, log to MLflow."""
    config = _load_config()
    models = _load_models()
    X_test, y_test = _load_test_data()

    all_metrics: dict = {}
    for model_key, model in models.items():
        metrics = _compute_metrics(model, X_test, y_test)
        result_key = _RESULT_KEYS[model_key]
        all_metrics[result_key] = metrics

        report = classification_report(
            y_test,
            model.predict(X_test),
            target_names=["Non-Diabetic", "Diabetic"],
        )
        logger.info(
            "classification_report",
            extra={"model": model_key, "report": report},
        )
        _save_confusion_matrix(model, X_test, y_test, model_key)
        logger.info("metrics_computed", extra={"model": model_key, **metrics})

    _save_roc_curves(models, X_test, y_test)

    # Merge new metrics into existing results.json (preserves top_shap_features)
    existing: dict = {}
    if _RESULTS_FILE.exists():
        with open(_RESULTS_FILE) as fh:
            existing = json.load(fh)
    existing.update(all_metrics)
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_FILE, "w") as fh:
        json.dump(existing, fh, indent=2)
    logger.info("results_json_updated", extra={"path": str(_RESULTS_FILE)})

    _log_to_mlflow(all_metrics, config)
    logger.info("evaluation_complete")


if __name__ == "__main__":
    main()
