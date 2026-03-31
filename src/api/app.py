"""Day 5 — Production FastAPI: /health, /predict (SHAP), /model_info."""

import json
import pathlib
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import joblib
import pandas as pd
import psutil
import yaml
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.data.schemas import PredictInput, PredictOutput
from src.exceptions import ModelNotFoundError, PredictionError
from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config (loaded once at import time)
# ---------------------------------------------------------------------------
with open("config/config.yaml") as _fh:
    _config = yaml.safe_load(_fh)

_MODELS_DIR = pathlib.Path("models")
_REPORTS_FILE = pathlib.Path("reports/results.json")

_FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# ---------------------------------------------------------------------------
# Shared application state — populated during lifespan startup
# ---------------------------------------------------------------------------
_state: dict = {
    "model": None,
    "imputer": None,
    "explainer": None,
    "start_time": None,
}

# ---------------------------------------------------------------------------
# Rate limiter (slowapi)
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Lifespan — load artifacts once; release on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load calibrated_xgb, imputer, and shap_explainer on startup."""
    _state["start_time"] = time.time()

    required = ("calibrated_xgb", "imputer", "shap_explainer")
    for name in required:
        p = _MODELS_DIR / f"{name}.joblib"
        if not p.exists():
            raise ModelNotFoundError(f"Required model artifact missing: {p}")

    _state["model"] = joblib.load(_MODELS_DIR / "calibrated_xgb.joblib")
    _state["imputer"] = joblib.load(_MODELS_DIR / "imputer.joblib")
    _state["explainer"] = joblib.load(_MODELS_DIR / "shap_explainer.joblib")

    logger.info(
        "startup_complete",
        extra={"artifacts": list(required)},
    )
    yield
    logger.info("shutdown")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="B2 Diabetes Classifier API",
    description=(
        "XGBoost + SHAP explainability REST endpoint "
        "for single-patient diabetes risk scoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Attach rate-limiter state and its 429 handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Prometheus /metrics endpoint
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ---------------------------------------------------------------------------
# Middleware (order matters: last added = outermost)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=_config["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],
)


@app.middleware("http")
async def check_content_length(request: Request, call_next):
    """Reject any request body larger than max_payload_mb (config)."""
    max_bytes = int(_config["api"]["max_payload_mb"]) * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_bytes:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "detail": (
                    f"Payload exceeds " f"{_config['api']['max_payload_mb']} MB limit."
                )
            },
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(PredictionError)
async def prediction_error_handler(
    request: Request, exc: PredictionError
) -> JSONResponse:
    """Map any PredictionError to HTTP 422 Unprocessable Entity."""
    logger.error("prediction_error", extra={"detail": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/v1/health", tags=["ops"])
async def health() -> dict:
    """
    Return API health status.

    Returns a dict with:
    - **status**: "ok" when the service is healthy.
    - **model_loaded**: True once calibrated_xgb is in memory.
    - **uptime_seconds**: seconds since the last startup.
    - **memory_mb**: resident set size of the current process.
    """
    proc = psutil.Process()
    start = _state["start_time"] or time.time()
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "uptime_seconds": round(time.time() - start, 2),
        "memory_mb": round(proc.memory_info().rss / 1024 / 1024, 2),
    }


@app.post(
    "/api/v1/predict",
    response_model=PredictOutput,
    tags=["inference"],
)
@limiter.limit(_config["api"]["rate_limit_predict"])
async def predict(request: Request, body: PredictInput) -> PredictOutput:
    """
    Score a single patient and return a SHAP-explained prediction.

    Applies the fitted MeanMedianImputer, runs the calibrated XGBoost
    classifier, and computes per-feature SHAP values via the serialised
    ExactExplainer.

    Args:
        body: PredictInput — 8 Pima diabetes features.

    Returns:
        PredictOutput with prediction (0/1), calibrated probability (0-1),
        human-readable label, per-feature SHAP values dict, and a trace_id
        for distributed tracing.

    Raises:
        PredictionError (HTTP 422): on any inference failure.
    """
    try:
        input_df = pd.DataFrame([body.dict()])
        input_imp = _state["imputer"].transform(input_df)

        pred = _state["model"].predict(input_imp)[0]
        prob = float(_state["model"].predict_proba(input_imp)[0][1])

        shap_out = _state["explainer"](input_imp)
        shap_raw = shap_out.values
        # ExactExplainer on a CalibratedClassifierCV returns shape
        # (1, n_features, n_classes) — take class-1 slice.
        if shap_raw.ndim == 3:
            shap_single = shap_raw[0, :, 1]
        else:
            shap_single = shap_raw[0]

        shap_dict = dict(zip(_FEATURE_NAMES, shap_single.tolist()))
        trace_id = str(uuid4())

        logger.info(
            "prediction_served",
            extra={
                "trace_id": trace_id,
                "prediction": int(pred),
                "prob": round(prob, 4),
            },
        )

        return PredictOutput(
            prediction=int(pred),
            probability=prob,
            label="Diabetic" if pred else "Non-Diabetic",
            shap_values=shap_dict,
            calibrated_probability=prob,
            trace_id=trace_id,
        )
    except PredictionError:
        raise
    except Exception as exc:
        logger.error("prediction_exception", extra={"error": str(exc)})
        raise PredictionError(str(exc)) from exc


@app.get("/api/v1/model_info", tags=["ops"])
async def model_info() -> dict:
    """
    Return model metadata and evaluation metrics.

    Reads reports/results.json (written by evaluate.py) and returns it
    alongside the registered model name from config.

    Returns a dict with:
    - **model**: name of the serving model ("calibrated_xgb").
    - **registered_model**: MLflow registered model name.
    - **results**: full contents of reports/results.json.
    """
    results: dict = {}
    if _REPORTS_FILE.exists():
        with open(_REPORTS_FILE) as fh:
            results = json.load(fh)

    return {
        "model": "calibrated_xgb",
        "registered_model": _config["mlflow"]["registered_model_name"],
        "results": results,
    }
