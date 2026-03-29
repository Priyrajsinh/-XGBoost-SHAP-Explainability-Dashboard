"""Day 0 scaffold smoke tests — must all pass before any training begins."""

import pytest


def test_diabetes_schema_exists():
    from src.data.validation import DIABETES_SCHEMA

    assert DIABETES_SCHEMA


def test_predict_input_rejects_zero_glucose():
    from src.data.schemas import PredictInput

    with pytest.raises(Exception):
        PredictInput(
            Pregnancies=1,
            Glucose=0,
            BloodPressure=70,
            SkinThickness=20,
            Insulin=80,
            BMI=25.0,
            DiabetesPedigreeFunction=0.5,
            Age=30,
        )


def test_exceptions_hierarchy():
    from src.exceptions import DataLoadError, ProjectBaseError

    assert issubclass(DataLoadError, ProjectBaseError)


def test_get_logger_returns_logger():
    import logging

    from src.logger import get_logger

    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"


def test_get_logger_no_duplicate_handlers():
    from src.logger import get_logger

    logger = get_logger("test_logger_dedup")
    get_logger("test_logger_dedup")
    assert len(logger.handlers) == 1


def test_predict_output_auto_trace_id():
    from src.data.schemas import PredictOutput

    out = PredictOutput(
        prediction=1,
        probability=0.8,
        label="Diabetic",
        shap_values={"Glucose": 0.3},
        calibrated_probability=0.75,
    )
    assert out.trace_id != ""


def test_predict_output_explicit_trace_id():
    from src.data.schemas import PredictOutput

    out = PredictOutput(
        prediction=0,
        probability=0.2,
        label="Non-Diabetic",
        shap_values={"BMI": -0.1},
        calibrated_probability=0.18,
        trace_id="fixed-id-123",
    )
    assert out.trace_id == "fixed-id-123"


def test_base_model_cannot_instantiate_directly():
    from src.models.base import BaseModel

    with pytest.raises(TypeError):
        BaseModel()  # type: ignore[abstract]


def test_base_model_concrete_subclass():
    import numpy as np
    import pandas as pd

    from src.models.base import BaseModel

    class DummyModel(BaseModel):
        def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
            pass

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros((len(X), 2))

        def save(self, path: str) -> None:
            pass

        def load(self, path: str) -> None:
            pass

    model = DummyModel()
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    model.fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    model.save("dummy.pkl")
    model.load("dummy.pkl")
    assert preds.shape == (3,)
    assert proba.shape == (3, 2)
