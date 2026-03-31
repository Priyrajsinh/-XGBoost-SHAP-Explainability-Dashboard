"""Gradio demo — self-contained: loads model + imputer, computes SHAP inline."""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import gradio as gr

# ---------------------------------------------------------------------------
# Load artifacts once at startup (files sit next to app.py in the Space)
# ---------------------------------------------------------------------------
_model = joblib.load("calibrated_xgb.joblib")
_imputer = joblib.load("imputer.joblib")

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

# PermutationExplainer: pure-Python, works on any callable, no numba needed.
# Background = zero vector; max_evals = 2*8+1 = 17 passes per request.
_background = shap.maskers.Independent(
    np.zeros((1, len(_FEATURE_NAMES))), max_samples=1
)
_explainer = shap.PermutationExplainer(_model.predict_proba, _background)

# Pima dataset ranges: (min, max, default)
_RANGES = {
    "Pregnancies":              (0,     17,    3),
    "Glucose":                  (44,    199,   117),
    "BloodPressure":            (24,    122,   72),
    "SkinThickness":            (0,     99,    23),
    "Insulin":                  (0,     846,   30),
    "BMI":                      (18.2,  67.1,  32.0),
    "DiabetesPedigreeFunction": (0.078, 2.42,  0.372),
    "Age":                      (21,    81,    29),
}


# ---------------------------------------------------------------------------
# Prediction + SHAP
# ---------------------------------------------------------------------------
def predict(
    pregnancies, glucose, blood_pressure, skin_thickness,
    insulin, bmi, dpf, age,
):
    """Run imputation → calibrated XGBoost → SHAP, return result + plot."""
    row = {
        "Pregnancies": float(pregnancies),
        "Glucose": float(glucose),
        "BloodPressure": float(blood_pressure),
        "SkinThickness": float(skin_thickness),
        "Insulin": float(insulin),
        "BMI": float(bmi),
        "DiabetesPedigreeFunction": float(dpf),
        "Age": float(age),
    }
    input_df = pd.DataFrame([row])
    input_imp = _imputer.transform(input_df)

    pred = int(_model.predict(input_imp)[0])
    prob = float(_model.predict_proba(input_imp)[0][1])

    label = "Diabetic" if pred == 1 else "Non-Diabetic"
    color = "#e63946" if pred == 1 else "#2a9d8f"
    result_text = f"**{label}** — {prob:.1%} probability of diabetes"

    # SHAP
    shap_out = _explainer(input_imp)
    shap_vals = shap_out.values[0, :, 1]  # class-1 slice

    sorted_idx = np.argsort(np.abs(shap_vals))
    feat_sorted = [_FEATURE_NAMES[i] for i in sorted_idx]
    vals_sorted = shap_vals[sorted_idx]
    bar_colors = ["#e63946" if v > 0 else "#2a9d8f" for v in vals_sorted]

    fig = go.Figure(
        go.Bar(
            x=vals_sorted,
            y=feat_sorted,
            orientation="h",
            marker_color=bar_colors,
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="black")
    fig.update_layout(
        title="SHAP values — feature contribution to this prediction",
        xaxis_title="SHAP value (red = pushes toward Diabetic)",
        height=360,
        margin=dict(l=160, r=20, t=50, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return result_text, fig


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def _slider(feat):
    lo, hi, default = _RANGES[feat]
    step = 1.0 if feat in ("Pregnancies", "Age") else 0.001
    return gr.Slider(float(lo), float(hi), value=float(default), step=step, label=feat)


with gr.Blocks(title="Diabetes Risk — XGBoost + SHAP") as demo:
    gr.Markdown(
        "## Diabetes Risk Classifier — XGBoost + SHAP\n"
        "Adjust the patient features and click **Predict**. "
        "The bar chart shows which features pushed the prediction toward or away from Diabetic."
    )
    with gr.Row():
        with gr.Column():
            s_preg = _slider("Pregnancies")
            s_gluc = _slider("Glucose")
            s_bp   = _slider("BloodPressure")
            s_skin = _slider("SkinThickness")
        with gr.Column():
            s_ins  = _slider("Insulin")
            s_bmi  = _slider("BMI")
            s_dpf  = _slider("DiabetesPedigreeFunction")
            s_age  = _slider("Age")

    btn = gr.Button("Predict", variant="primary")

    with gr.Row():
        result = gr.Markdown()
    with gr.Row():
        plot = gr.Plot()

    btn.click(
        fn=predict,
        inputs=[s_preg, s_gluc, s_bp, s_skin, s_ins, s_bmi, s_dpf, s_age],
        outputs=[result, plot],
    )

if __name__ == "__main__":
    demo.launch()
