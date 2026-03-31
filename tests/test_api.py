"""Tests for src/api/app.py — FastAPI health, predict, model_info."""

import pathlib
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixture — mock model state so tests never touch disk
# ---------------------------------------------------------------------------


def _make_mocks():
    """Return (mock_model, mock_imputer, mock_explainer_instance)."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

    mock_imputer = MagicMock()
    mock_imputer.transform.return_value = np.zeros((1, 8))

    # TreeExplainer returns (1, n_features) — 2-D (no class axis for binary)
    mock_shap_out = MagicMock()
    mock_shap_out.values = np.zeros((1, 8))
    mock_explainer_instance = MagicMock()
    mock_explainer_instance.return_value = mock_shap_out

    return mock_model, mock_imputer, mock_explainer_instance


def _patched_exists(self):
    """Return True for model .joblib paths; delegate others to os.path."""
    import os

    s = str(self)
    if "models" in s and s.endswith(".joblib"):
        return True
    return os.path.exists(s)


@pytest.fixture(scope="module")
def api_client():
    """TestClient with lifespan running against mocked artifacts."""
    import src.api.app as app_mod

    mock_model, mock_imputer, mock_explainer_instance = _make_mocks()

    with ExitStack() as stack:
        # Lifespan now calls joblib.load twice (model + imputer only)
        mock_load = stack.enter_context(patch("src.api.app.joblib.load"))
        mock_load.side_effect = [mock_model, mock_imputer]
        # TreeExplainer is rebuilt at startup — return our mock instance
        mock_tree_exp = stack.enter_context(patch("src.api.app.shap.TreeExplainer"))
        mock_tree_exp.return_value = mock_explainer_instance
        stack.enter_context(patch.object(pathlib.Path, "exists", _patched_exists))
        client = stack.enter_context(TestClient(app_mod.app))
        yield client


# ---------------------------------------------------------------------------
# /api/v1/health
# ---------------------------------------------------------------------------


def test_health_returns_200(api_client):
    resp = api_client.get("/api/v1/health")
    assert resp.status_code == 200


def test_health_response_shape(api_client):
    data = api_client.get("/api/v1/health").json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data
    assert "memory_mb" in data


# ---------------------------------------------------------------------------
# POST /api/v1/predict
# ---------------------------------------------------------------------------

_VALID_PAYLOAD = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 25.5,
    "DiabetesPedigreeFunction": 0.3,
    "Age": 28,
}


def test_predict_returns_200(api_client):
    resp = api_client.post("/api/v1/predict", json=_VALID_PAYLOAD)
    assert resp.status_code == 200


def test_predict_response_fields(api_client):
    data = api_client.post("/api/v1/predict", json=_VALID_PAYLOAD).json()
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0
    assert data["label"] in ("Diabetic", "Non-Diabetic")
    assert len(data["shap_values"]) == 8
    assert all(isinstance(v, float) for v in data["shap_values"].values())
    assert len(data["trace_id"]) == 36  # UUID4 format


def test_predict_shap_keys_match_features(api_client):
    expected = {
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    }
    data = api_client.post("/api/v1/predict", json=_VALID_PAYLOAD).json()
    assert set(data["shap_values"].keys()) == expected


def test_predict_rejects_zero_glucose(api_client):
    bad = {**_VALID_PAYLOAD, "Glucose": 0}
    resp = api_client.post("/api/v1/predict", json=bad)
    # Pydantic validator raises a validation error → 422
    assert resp.status_code == 422


def test_predict_rejects_missing_field(api_client):
    incomplete = {k: v for k, v in _VALID_PAYLOAD.items() if k != "BMI"}
    resp = api_client.post("/api/v1/predict", json=incomplete)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/model_info
# ---------------------------------------------------------------------------


def test_model_info_returns_200(api_client):
    resp = api_client.get("/api/v1/model_info")
    assert resp.status_code == 200


def test_model_info_response_shape(api_client):
    data = api_client.get("/api/v1/model_info").json()
    assert data["model"] == "calibrated_xgb"
    assert "registered_model" in data
    assert "results" in data
