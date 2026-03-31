"""Day 5 — Gradio demo: 8 sliders → POST /api/v1/predict → label + SHAP."""

import pathlib

import gradio as gr
import httpx
import pandas as pd
import yaml

from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config & slider bounds from training data
# ---------------------------------------------------------------------------
with open("config/config.yaml") as _fh:
    _config = yaml.safe_load(_fh)

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

_API_URL = "http://localhost:8000/api/v1/predict"


def _slider_bounds() -> dict:
    """Return {feature: (min, max, median)} from train.csv."""
    p = pathlib.Path("data/processed/train.csv")
    df = pd.read_csv(p)
    return {
        feat: (
            float(df[feat].min()),
            float(df[feat].max()),
            float(df[feat].median()),
        )
        for feat in _FEATURE_NAMES
    }


_BOUNDS = _slider_bounds()


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------
def predict(
    pregnancies: float,
    glucose: float,
    blood_pressure: float,
    skin_thickness: float,
    insulin: float,
    bmi: float,
    dpf: float,
    age: float,
) -> str:
    """POST to /api/v1/predict and return formatted result string."""
    payload = {
        "Pregnancies": int(pregnancies),
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": int(age),
    }
    try:
        resp = httpx.post(_API_URL, json=payload, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
    except httpx.RequestError as exc:
        logger.error("api_request_failed", extra={"error": str(exc)})
        return f"Error: could not reach API — {exc}"
    except httpx.HTTPStatusError as exc:
        return f"API error {exc.response.status_code}: {exc.response.text}"

    label = data["label"]
    prob = data["calibrated_probability"]
    shap = data["shap_values"]

    # Top-3 features by absolute SHAP magnitude
    top3 = sorted(shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    shap_lines = "\n".join(f"  {feat}: {val:+.4f}" for feat, val in top3)
    logger.info(
        "demo_prediction",
        extra={"label": label, "prob": round(prob, 4)},
    )
    return (
        f"Prediction : {label}\n"
        f"Probability: {prob:.1%}\n\n"
        f"Top-3 SHAP features:\n{shap_lines}"
    )


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------
def _make_slider(feat: str) -> gr.Slider:
    lo, hi, med = _BOUNDS[feat]
    step = 1.0 if feat in ("Pregnancies", "Age") else 0.001
    return gr.Slider(
        minimum=lo,
        maximum=hi,
        value=med,
        step=step,
        label=feat,
    )


with gr.Blocks(title="Diabetes Risk — XGBoost + SHAP") as demo:
    gr.Markdown("## Diabetes Risk Classifier\nAdjust features and click **Predict**.")
    with gr.Row():
        with gr.Column():
            sliders = [_make_slider(f) for f in _FEATURE_NAMES[:4]]
        with gr.Column():
            sliders += [_make_slider(f) for f in _FEATURE_NAMES[4:]]
    btn = gr.Button("Predict", variant="primary")
    output = gr.Textbox(label="Result", lines=7)
    btn.click(fn=predict, inputs=sliders, outputs=output)

if __name__ == "__main__":
    demo.launch(share=True)
