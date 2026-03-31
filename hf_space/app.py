"""Hugging Face Space — minimal Gradio UI that calls the deployed API."""

import os

import gradio as gr
import httpx

# Replace with the actual deployed API URL before pushing to HF Spaces.
_API_URL = os.environ.get(
    "API_URL",
    "https://YOUR_DEPLOYED_API_URL/api/v1/predict",
)

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

# Pima dataset ranges (min, max, default)
_SLIDER_RANGES = {
    "Pregnancies":            (0,     17,    3),
    "Glucose":                (44,    199,   117),
    "BloodPressure":          (24,    122,   72),
    "SkinThickness":          (0,     99,    23),
    "Insulin":                (0,     846,   30),
    "BMI":                    (18.2,  67.1,  32.0),
    "DiabetesPedigreeFunction": (0.078, 2.42, 0.372),
    "Age":                    (21,    81,    29),
}


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
    """Send a prediction request to the deployed API."""
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
        resp = httpx.post(_API_URL, json=payload, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
    except httpx.RequestError as exc:
        return f"Error: could not reach API — {exc}"
    except httpx.HTTPStatusError as exc:
        return f"API error {exc.response.status_code}: {exc.response.text}"

    label = data["label"]
    prob = data["calibrated_probability"]
    shap = data["shap_values"]

    top3 = sorted(shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    shap_lines = "\n".join(f"  {feat}: {val:+.4f}" for feat, val in top3)
    return (
        f"Prediction : {label}\n"
        f"Probability: {prob:.1%}\n\n"
        f"Top-3 SHAP features:\n{shap_lines}"
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def _make_slider(feat: str) -> gr.Slider:
    lo, hi, default = _SLIDER_RANGES[feat]
    step = 1.0 if feat in ("Pregnancies", "Age") else 0.001
    return gr.Slider(
        minimum=float(lo),
        maximum=float(hi),
        value=float(default),
        step=step,
        label=feat,
    )


with gr.Blocks(title="Diabetes Risk — XGBoost + SHAP") as demo:
    gr.Markdown(
        "## Diabetes Risk Classifier\n"
        "Adjust features and click **Predict** to score this patient."
    )
    with gr.Row():
        with gr.Column():
            sliders = [_make_slider(f) for f in _FEATURE_NAMES[:4]]
        with gr.Column():
            sliders += [_make_slider(f) for f in _FEATURE_NAMES[4:]]
    btn = gr.Button("Predict", variant="primary")
    output = gr.Textbox(label="Result", lines=7)
    btn.click(fn=predict, inputs=sliders, outputs=output)

if __name__ == "__main__":
    demo.launch()
