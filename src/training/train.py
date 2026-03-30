"""Day-2 training pipeline: Optuna HPO → XGBoost + LightGBM → MLflow Registry."""

import argparse
import pathlib

import joblib
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.exceptions import ConfigError, ModelNotFoundError
from src.logger import get_logger

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


def _load_config(config_path: str) -> dict:
    path = pathlib.Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config not found: {path}")
    with open(path) as fh:
        return yaml.safe_load(fh)


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test CSVs produced by Day-1 preprocessing."""
    for name in ("train", "val", "test"):
        p = _PROCESSED_DIR / f"{name}.csv"
        if not p.exists():
            raise ModelNotFoundError(f"Processed split missing: {p}")
    train_df = pd.read_csv(_PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(_PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(_PROCESSED_DIR / "test.csv")
    logger.info(
        "Splits loaded",
        extra={"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    )
    return train_df, val_df, test_df


def run_optuna_study(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
) -> optuna.Study:
    """Run Optuna TPE study to tune XGBoost hyperparameters.

    Results are persisted to SQLite so optuna-dashboard can replay trials live.
    """
    X_train = train_df[_FEATURE_COLS].values
    y_train = train_df["Outcome"].values
    X_val = val_df[_FEATURE_COLS].values
    y_val = val_df["Outcome"].values

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # as_posix() ensures forward slashes on Windows for SQLAlchemy
    db_path = (_MODELS_DIR / "optuna.db").as_posix()
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name="xgb-diabetes",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2),
        }
        # use_label_encoder removed in XGBoost 2.x
        model = XGBClassifier(**params, random_state=42, eval_metric="logloss")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return f1_score(y_val, model.predict(X_val), average="weighted")

    study.optimize(
        objective,
        n_trials=config["optuna"]["n_trials"],
        show_progress_bar=True,
    )
    logger.info(
        "Optuna study complete",
        extra={"best_value": study.best_value, "best_params": study.best_params},
    )
    return study


def train_final_models(
    best_params: dict,
    train_df: pd.DataFrame,
    config: dict,
) -> tuple[XGBClassifier, LGBMClassifier]:
    """Train XGBoost (Optuna best params) and LightGBM (config params)."""
    X_train = train_df[_FEATURE_COLS].values
    y_train = train_df["Outcome"].values

    # Label sanity check BEFORE any fit
    label_counts = train_df["Outcome"].value_counts()
    logger.info(
        "Labels",
        extra={
            "class_0": int(label_counts.get(0, 0)),
            "class_1": int(label_counts.get(1, 0)),
        },
    )
    assert set(train_df["Outcome"].unique()) == {0, 1}, "Unexpected labels"
    assert train_df["Outcome"].isna().sum() == 0, "NaN in labels before training"

    xgb_model = XGBClassifier(**best_params, random_state=42, eval_metric="logloss")
    lgbm_cfg = config["model"]["lgbm"]
    lgbm_model = LGBMClassifier(
        n_estimators=lgbm_cfg["n_estimators"],
        max_depth=lgbm_cfg["max_depth"],
        learning_rate=lgbm_cfg["learning_rate"],
        num_leaves=lgbm_cfg["num_leaves"],
        subsample=lgbm_cfg["subsample"],
        colsample_bytree=lgbm_cfg["colsample_bytree"],
        random_state=lgbm_cfg["random_state"],
        verbosity=lgbm_cfg["verbosity"],
    )

    xgb_model.fit(X_train, y_train)
    lgbm_model.fit(X_train, y_train)
    logger.info("Models trained", extra={"xgb": "done", "lgbm": "done"})

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_model, _MODELS_DIR / "xgb_model.joblib")
    joblib.dump(lgbm_model, _MODELS_DIR / "lgbm_model.joblib")
    logger.info("Models saved", extra={"dir": str(_MODELS_DIR)})
    return xgb_model, lgbm_model


def _register_to_mlflow(
    xgb_model: XGBClassifier,
    study: optuna.Study,
    config: dict,
) -> None:
    """Log XGBoost + best params to MLflow; register to Model Registry at Staging."""
    mlflow_cfg = config["mlflow"]
    registered_name = mlflow_cfg["registered_model_name"]

    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    with mlflow.start_run(run_name="optuna_xgb") as run:
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_f1", study.best_value)
        mlflow.sklearn.log_model(xgb_model, "xgb_model")
        run_id = run.info.run_id

    mlflow.register_model(f"runs:/{run_id}/xgb_model", registered_name)

    client = mlflow.tracking.MlflowClient()
    latest = client.get_latest_versions(registered_name)[0]
    client.transition_model_version_stage(registered_name, latest.version, "Staging")
    logger.info(
        "MLflow model registered",
        extra={"run_id": run_id, "version": latest.version, "stage": "Staging"},
    )


def main(config_path: str = "config/config.yaml") -> None:
    """End-to-end Day-2 pipeline: load splits → HPO → train → register."""
    config = _load_config(config_path)
    train_df, val_df, _ = _load_splits()

    study = run_optuna_study(train_df, val_df, config)
    xgb_model, _ = train_final_models(study.best_params, train_df, config)
    _register_to_mlflow(xgb_model, study, config)
    logger.info("Day 2 training pipeline complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day-2 training pipeline")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config/config.yaml",
    )
    args = parser.parse_args()
    main(args.config)
