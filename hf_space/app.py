"""Gradio demo — 3 tabs: Predict + Global SHAP + Model Performance."""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import gradio as gr

# ---------------------------------------------------------------------------
# Load artifacts once at startup
# ---------------------------------------------------------------------------
_model = joblib.load("calibrated_xgb.joblib")
_imputer = joblib.load("imputer.joblib")
_shap_vals = np.load("shap_values.npy")  # (n_test, 8) — class-1 SHAP

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

_background = shap.maskers.Independent(
    np.zeros((1, len(_FEATURE_NAMES))), max_samples=1
)
_explainer = shap.PermutationExplainer(_model.predict_proba, _background)

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
# Tab 1 — Predict
# ---------------------------------------------------------------------------
def predict(pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age):
    """Run imputation → calibrated XGBoost → SHAP, return result + bar chart."""
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

    result_md = f"### {'🔴' if pred else '🟢'} {label} — {prob:.1%} probability of diabetes"

    shap_out = _explainer(input_imp)
    shap_arr = shap_out.values[0, :, 1]

    sorted_idx = np.argsort(np.abs(shap_arr))
    feat_sorted = [_FEATURE_NAMES[i] for i in sorted_idx]
    vals_sorted = shap_arr[sorted_idx]
    bar_colors = ["#e63946" if v > 0 else "#2a9d8f" for v in vals_sorted]

    fig = go.Figure(go.Bar(
        x=vals_sorted, y=feat_sorted, orientation="h",
        marker_color=bar_colors,
    ))
    fig.add_vline(x=0, line_width=1, line_color="black")
    fig.update_layout(
        title="SHAP — feature contribution to this prediction",
        xaxis_title="SHAP value  (red = pushes toward Diabetic, teal = away)",
        height=380,
        margin=dict(l=180, r=20, t=50, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return result_md, fig


def _slider(feat):
    lo, hi, default = _RANGES[feat]
    step = 1.0 if feat in ("Pregnancies", "Age") else 0.001
    return gr.Slider(float(lo), float(hi), value=float(default), step=step, label=feat)


# ---------------------------------------------------------------------------
# Tab 2 — Global SHAP  (precomputed from shap_values.npy)
# ---------------------------------------------------------------------------
def _global_bar_fig():
    mean_abs = np.abs(_shap_vals).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)
    feat_sorted = [_FEATURE_NAMES[i] for i in sorted_idx]
    vals_sorted = mean_abs[sorted_idx]

    fig = go.Figure(go.Bar(
        x=vals_sorted, y=feat_sorted, orientation="h",
        marker=dict(color=vals_sorted, colorscale="Blues"),
    ))
    fig.update_layout(
        title="Global Feature Importance — mean |SHAP| across test set",
        xaxis_title="Mean |SHAP value|",
        height=380,
        margin=dict(l=180, r=20, t=50, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def _dot_plot_fig():
    """Beeswarm-style dot plot: one dot per test sample, coloured by feature value."""
    figs = go.Figure()
    for i, feat in enumerate(_FEATURE_NAMES):
        col_vals = _shap_vals[:, i]
        # colour by rank of the feature value (we only have SHAP, not raw values)
        figs.add_trace(go.Scatter(
            x=col_vals,
            y=[feat] * len(col_vals),
            mode="markers",
            marker=dict(
                size=4,
                color=col_vals,
                colorscale="RdBu_r",
                opacity=0.6,
            ),
            name=feat,
            showlegend=False,
        ))
    figs.add_vline(x=0, line_width=1, line_color="black")
    figs.update_layout(
        title="SHAP Dot Plot — distribution across test set",
        xaxis_title="SHAP value",
        height=420,
        margin=dict(l=180, r=20, t=50, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return figs


_GLOBAL_BAR = _global_bar_fig()
_DOT_PLOT = _dot_plot_fig()


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Diabetes Risk — XGBoost + SHAP") as demo:
    gr.Markdown(
        "# Diabetes Risk Classifier — XGBoost + SHAP\n"
        "XGBoost trained on the Pima Indians dataset with SHAP explainability. "
        "Optuna HPO · Calibrated probabilities · FastAPI backend."
    )

    with gr.Tabs():

        # ── Tab 1: Predict ───────────────────────────────────────────────
        with gr.Tab("🔮 Predict"):
            gr.Markdown("Adjust the patient features and click **Predict**.")
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
            result_md = gr.Markdown()
            shap_plot = gr.Plot()

            btn.click(
                fn=predict,
                inputs=[s_preg, s_gluc, s_bp, s_skin, s_ins, s_bmi, s_dpf, s_age],
                outputs=[result_md, shap_plot],
            )

        # ── Tab 2: Global SHAP ───────────────────────────────────────────
        with gr.Tab("🐝 Global SHAP"):
            gr.Markdown(
                "**Mean |SHAP| bar chart** (top) ranks features by average impact "
                "across the entire test set.  \n"
                "**Dot plot** (bottom) shows the SHAP value distribution per feature — "
                "red = pushed toward Diabetic, blue = away."
            )
            gr.Plot(value=_GLOBAL_BAR)
            gr.Plot(value=_DOT_PLOT)

        # ── Tab 3: Model Performance ──────────────────────────────────────
        with gr.Tab("📊 Model Performance"):
            gr.Markdown(
                "ROC curves and SHAP beeswarm for XGBoost · LightGBM · Calibrated XGBoost."
            )
            gr.Image(value="roc_curves.png", label="ROC Curves — 3-model comparison")
            gr.Image(value="shap_beeswarm.png", label="SHAP Beeswarm — test set")

if __name__ == "__main__":
    demo.launch()
