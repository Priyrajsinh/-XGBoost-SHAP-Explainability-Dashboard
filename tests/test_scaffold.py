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
