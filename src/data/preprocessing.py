"""Imputation and stratified train/val/test split for the Pima dataset.

Exported public API
-------------------
impute_and_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    Returns (train_df, val_df, test_df) after median imputation fitted on
    train only.  Persists processed CSVs and the fitted imputer to disk.
"""

import pathlib

import joblib
import pandas as pd
import yaml
from feature_engine.imputation import MeanMedianImputer
from sklearn.model_selection import StratifiedShuffleSplit

from src.logger import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = pathlib.Path("config/config.yaml")
_PROCESSED_DIR = pathlib.Path("data/processed")
_MODELS_DIR = pathlib.Path("models")
_IMPUTER_PATH = _MODELS_DIR / "imputer.joblib"

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


def _load_config() -> dict:
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def impute_and_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split then median-impute Pima features.

    Order of operations (critical — imputer must never see val/test labels)
    ------------------------------------------------------------------------
    1. Stratified 70 / 15 / 15 split on ``Outcome`` using two successive
       ``StratifiedShuffleSplit`` passes. Zero overlap is asserted.
    2. Fit ``MeanMedianImputer`` on *train* features only.
    3. Transform train, val, and test separately.
    4. Log split sizes and per-split class distribution.
    5. Persist ``data/processed/{train,val,test}.csv`` and
       ``models/imputer.joblib``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as returned by ``load_pima()``, with NaN values for
        zero-imputed columns.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)`` — all features imputed, Outcome intact.

    Raises
    ------
    AssertionError
        If train / val / test index sets overlap.
    """
    config = _load_config()
    seed: int = config["data"]["seed"]
    val_split: float = config["data"]["val_split"]
    test_split: float = config["data"]["test_split"]
    zero_as_nan_cols: list[str] = config["data"]["zero_as_nan_cols"]

    X = df[_FEATURE_COLS]
    y = df["Outcome"]

    # --- Step 1: stratified splits (BEFORE any imputer.fit) ---
    # First pass: carve out the test set
    test_frac = test_split  # 0.15 of full dataset
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(X, y))

    # Second pass: carve val from the remaining train+val pool
    # val_split refers to fraction of the *full* dataset, so we rescale
    val_frac_of_trainval = val_split / (1.0 - test_split)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac_of_trainval, random_state=seed
    )
    trainval_X = X.iloc[trainval_idx]
    trainval_y = y.iloc[trainval_idx]
    train_sub_idx, val_sub_idx = next(sss2.split(trainval_X, trainval_y))

    # Map back to original dataframe indices
    train_idx = trainval_idx[train_sub_idx]
    val_idx = trainval_idx[val_sub_idx]

    # Sanity: zero overlap
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap detected"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap detected"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap detected"

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    logger.info(
        "Splits created",
        extra={
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    )

    # --- Step 2: label sanity check on train ---
    label_counts = train_df["Outcome"].value_counts()
    logger.info(
        "Labels",
        extra={
            "class_0": int(label_counts.get(0, 0)),
            "class_1": int(label_counts.get(1, 0)),
        },
    )
    assert set(train_df["Outcome"].unique()) == {0, 1}, "Unexpected labels"
    assert train_df["Outcome"].isna().sum() == 0, "NaN in labels"

    # --- Step 3: fit imputer on train only ---
    imputer = MeanMedianImputer(
        imputation_method="median",
        variables=zero_as_nan_cols,  # type: ignore[arg-type]
    )
    imputer.fit(train_df[_FEATURE_COLS])
    logger.info("Imputer fitted on train set", extra={"variables": zero_as_nan_cols})

    # --- Step 4: transform each split separately ---
    train_df[_FEATURE_COLS] = imputer.transform(train_df[_FEATURE_COLS])
    val_df[_FEATURE_COLS] = imputer.transform(val_df[_FEATURE_COLS])
    test_df[_FEATURE_COLS] = imputer.transform(test_df[_FEATURE_COLS])

    # --- Step 5: log class distributions per split ---
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split["Outcome"].value_counts().to_dict()
        logger.info(
            f"{name} class distribution",
            extra={"class_0": int(dist.get(0, 0)), "class_1": int(dist.get(1, 0))},
        )

    # --- Step 6: persist ---
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(_PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(_PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(_PROCESSED_DIR / "test.csv", index=False)
    logger.info("Processed CSVs saved", extra={"dir": str(_PROCESSED_DIR)})

    joblib.dump(imputer, _IMPUTER_PATH)
    logger.info("Imputer saved", extra={"path": str(_IMPUTER_PATH)})

    return train_df, val_df, test_df
