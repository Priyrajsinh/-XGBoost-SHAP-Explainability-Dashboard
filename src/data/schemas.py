"""Pydantic request/response schemas for the FastAPI prediction endpoint."""
import uuid

from pydantic import BaseModel, validator


class PredictInput(BaseModel):
    """Input schema for /api/v1/predict."""

    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

    @validator("Glucose")
    def glucose_positive(cls, v: float) -> float:
        """Glucose of 0 is biologically impossible — reject it."""
        if v <= 0:
            raise ValueError("Glucose must be > 0")
        return v


class PredictOutput(BaseModel):
    """Output schema for /api/v1/predict."""

    prediction: int
    probability: float
    label: str
    shap_values: dict[str, float]
    calibrated_probability: float
    trace_id: str = ""

    def __init__(self, **data):
        if not data.get("trace_id"):
            data["trace_id"] = str(uuid.uuid4())
        super().__init__(**data)
