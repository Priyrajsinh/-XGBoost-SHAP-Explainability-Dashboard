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
