"""Load the Pima Indians Diabetes dataset from the canonical URL.

Exported public API
-------------------
load_pima() -> pd.DataFrame
    Returns the raw dataframe with zeros replaced by NaN for physiologically
    impossible columns, validates against DIABETES_SCHEMA, computes a SHA-256
    checksum, and persists data/raw/pima.csv + data/raw/checksums.json.
"""

import hashlib
import json
import pathlib

import numpy as np
import pandas as pd
import yaml

from src.data.validation import DIABETES_SCHEMA
from src.exceptions import DataLoadError, DataValidationError
from src.logger import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = pathlib.Path("config/config.yaml")
_RAW_DIR = pathlib.Path("data/raw")
_PIMA_CSV = _RAW_DIR / "pima.csv"
_CHECKSUMS_JSON = _RAW_DIR / "checksums.json"

_PIMA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master"
    "/pima-indians-diabetes.data.csv"
)
_COLUMNS = [
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


def _load_config() -> dict:
    """Read config/config.yaml and return the parsed dict."""
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def load_pima() -> pd.DataFrame:
    """Download and pre-process the Pima Indians Diabetes dataset.

    Steps
    -----
    1. Read CSV from the canonical GitHub URL.
    2. Replace physiologically impossible zeros with NaN for the columns
       listed in ``config['data']['zero_as_nan_cols']``.
    3. Compute a SHA-256 checksum of the CSV representation and save it to
       ``data/raw/checksums.json``.
    4. Validate the dataframe against ``DIABETES_SCHEMA`` (NaN filled with 0
       for the schema check only).
    5. Log total row count, per-column NaN counts, and Outcome distribution.
    6. Save the dataframe to ``data/raw/pima.csv``.

    Returns
    -------
    pd.DataFrame
        The processed dataframe with NaN values in zero-as-NaN columns.

    Raises
    ------
    DataLoadError
        If the CSV cannot be fetched or parsed.
    DataValidationError
        If the dataframe fails pandera schema validation.
    """
    config = _load_config()
    zero_as_nan_cols: list[str] = config["data"]["zero_as_nan_cols"]

    logger.info("Fetching Pima dataset", extra={"url": _PIMA_URL})
    try:
        df = pd.read_csv(_PIMA_URL, names=_COLUMNS)
    except Exception as exc:
        raise DataLoadError(f"Failed to fetch Pima CSV: {exc}") from exc

    # Enforce dtypes to match DIABETES_SCHEMA regardless of CSV source
    _int_cols = ["Pregnancies", "Age", "Outcome"]
    _float_cols = [c for c in _COLUMNS if c not in _int_cols]
    for col in _int_cols:
        df[col] = df[col].astype(int)
    for col in _float_cols:
        df[col] = df[col].astype(float)

    # Replace biologically impossible zeros with NaN
    for col in zero_as_nan_cols:
        n_zeros = (df[col] == 0).sum()
        df[col] = df[col].replace(0, np.nan)
        logger.info(
            "Replaced zeros with NaN", extra={"column": col, "count": int(n_zeros)}
        )

    # SHA-256 over the CSV string (deterministic column order)
    csv_bytes = df.to_csv(index=False).encode()
    checksum = hashlib.sha256(csv_bytes).hexdigest()
    logger.info("SHA-256 checksum computed", extra={"sha256": checksum})

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CHECKSUMS_JSON, "w") as fh:
        json.dump({"pima.csv": checksum}, fh, indent=2)
    logger.info("Checksum saved", extra={"path": str(_CHECKSUMS_JSON)})

    # Validate schema (fill NaN with 0 only for the schema check)
    try:
        DIABETES_SCHEMA.validate(df.fillna(0))
    except Exception as exc:
        raise DataValidationError(f"Schema validation failed: {exc}") from exc

    # Diagnostic logging
    logger.info("Dataset loaded", extra={"total_rows": len(df)})
    nan_counts = df.isna().sum().to_dict()
    logger.info("NaN counts per column", extra=nan_counts)
    outcome_dist = df["Outcome"].value_counts().to_dict()
    logger.info(
        "Outcome distribution",
        extra={
            "class_0": int(outcome_dist.get(0, 0)),
            "class_1": int(outcome_dist.get(1, 0)),
        },
    )

    df.to_csv(_PIMA_CSV, index=False)
    logger.info("Raw CSV saved", extra={"path": str(_PIMA_CSV)})

    return df
