"""Pandera schema for the Pima Indians Diabetes dataset.

Call DIABETES_SCHEMA.validate(df) BEFORE any train/test split.
"""
import pandera as pa
from pandera import Check, Column, DataFrameSchema

DIABETES_SCHEMA = DataFrameSchema(
    {
        "Pregnancies": Column(int, Check.ge(0)),
        "Glucose": Column(float, Check.gt(0)),
        "BloodPressure": Column(float, Check.ge(0)),
        "SkinThickness": Column(float, Check.ge(0)),
        "Insulin": Column(float, Check.ge(0)),
        "BMI": Column(float, Check.ge(0)),
        "DiabetesPedigreeFunction": Column(float, Check.gt(0)),
        "Age": Column(int, Check.gt(0)),
        "Outcome": Column(int, Check.isin([0, 1])),
    },
    strict=True,
)
