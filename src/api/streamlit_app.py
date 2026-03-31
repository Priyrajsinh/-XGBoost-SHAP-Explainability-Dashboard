"""Day 4 — Streamlit 4-tab dashboard: Predict, Global SHAP, Dependence, HPO."""

import json

import joblib
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Training-data min/max for sliders (hard-coded from data/processed/train.csv)
SLIDER_BOUNDS = {
    "Pregnancies": (0, 17, 3),
    "Glucose": (56, 199, 118),
    "BloodPressure": (24, 122, 72),
    "SkinThickness": (7, 99, 28),
    "Insulin": (15, 744, 125),
    "BMI": (18.2, 67.1, 32.0),
    "DiabetesPedigreeFunction": (0.084, 2.329, 0.378),
    "Age": (21, 72, 29),
}


# ---------------------------------------------------------------------------
# Cached loaders — never reloaded on interaction
# ---------------------------------------------------------------------------
@st.cache_data
def load_all():
    model = joblib.load("models/calibrated_xgb.joblib")
    imputer = joblib.load("models/imputer.joblib")
    explainer = joblib.load("models/shap_explainer.joblib")
    shap_vals = np.load("models/shap_values.npy")
    results = json.load(open("reports/results.json"))
    X_test = pd.read_csv("data/processed/test.csv").drop("Outcome", axis=1)
    return model, imputer, explainer, shap_vals, results, X_test


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Diabetes XGBoost Dashboard", layout="wide")
st.title("Diabetes Risk Classifier — XGBoost + SHAP")

model, imputer, explainer, shap_vals, results, X_test = load_all()

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Predict", "🐝 Global SHAP", "📈 Feature Dependence", "⚡ HPO History"]
)

# ===========================================================================
# TAB 1 — Predict
# ===========================================================================
with tab1:
    st.subheader("Single-Patient Prediction")

    col_left, col_right = st.columns(2)
    slider_values = {}

    for i, feat in enumerate(FEATURE_NAMES):
        lo, hi, default = SLIDER_BOUNDS[feat]
        col = col_left if i < 4 else col_right
        if isinstance(lo, float) or isinstance(hi, float):
            slider_values[feat] = col.slider(
                feat, float(lo), float(hi), float(default), step=0.001
            )
        else:
            slider_values[feat] = col.slider(feat, int(lo), int(hi), int(default))

    if st.button("Predict", type="primary"):
        input_df = pd.DataFrame([slider_values])
        input_imp = imputer.transform(input_df)

        pred = model.predict(input_imp)[0]
        prob = model.predict_proba(input_imp)[0][1]

        label = "Diabetic" if pred == 1 else "Non-Diabetic"
        color = "red" if pred == 1 else "green"
        st.markdown(
            f"**Prediction:** :{color}[{label}]"
            f" &nbsp;|&nbsp; **Probability:** `{prob:.1%}`"
        )

        shap_single = explainer(input_imp)
        # ExactExplainer returns shape (1, n_features, n_classes) — take class-1 slice
        sv_vals = shap_single.values[0, :, 1].tolist()
        base_val = float(shap_single.base_values[0, 1])

        fig = go.Figure(
            go.Waterfall(
                name="SHAP",
                orientation="h",
                y=FEATURE_NAMES,
                x=sv_vals,
                base=base_val,
                connector={"line": {"color": "rgb(63,63,63)"}},
                decreasing={"marker": {"color": "steelblue"}},
                increasing={"marker": {"color": "crimson"}},
            )
        )
        fig.update_layout(
            title="SHAP Waterfall — this prediction",
            height=400,
            margin={"l": 160},
        )
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 2 — Global SHAP
# ===========================================================================
with tab2:
    st.subheader("Global Feature Importance (mean |SHAP|)")

    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:20]

    fig = px.bar(
        x=mean_abs[top_idx],
        y=np.array(FEATURE_NAMES)[top_idx],
        orientation="h",
        color=mean_abs[top_idx],
        color_continuous_scale="Blues",
        labels={"x": "Mean |SHAP value|", "y": "Feature"},
    )
    fig.update_layout(coloraxis_showscale=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Comparison — XGBoost vs LightGBM")
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Weighted F1", "AUC-ROC", "Brier Score"],
            "XGBoost": [
                results["xgboost"]["accuracy"],
                results["xgboost"]["weighted_f1"],
                results["xgboost"]["auc_roc"],
                results["xgboost"]["brier_score"],
            ],
            "LightGBM": [
                results["lightgbm"]["accuracy"],
                results["lightgbm"]["weighted_f1"],
                results["lightgbm"]["auc_roc"],
                results["lightgbm"]["brier_score"],
            ],
        }
    )
    st.dataframe(metrics_df.set_index("Metric"), use_container_width=True)

# ===========================================================================
# TAB 3 — Feature Dependence
# ===========================================================================
with tab3:
    st.subheader("SHAP Dependence Plot")

    dep_features = config["shap"]["dependence_features"]
    feat_sel = st.selectbox("Feature", dep_features)
    feat_idx = FEATURE_NAMES.index(feat_sel)

    fig = px.scatter(
        x=X_test[feat_sel],
        y=shap_vals[:, feat_idx],
        color=X_test["Glucose"],
        color_continuous_scale="Viridis",
        labels={"x": feat_sel, "y": "SHAP value", "color": "Glucose"},
        title=f"SHAP Dependence: {feat_sel} (coloured by Glucose)",
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 4 — HPO History
# ===========================================================================
with tab4:
    st.subheader("Optuna HPO Convergence")

    study = optuna.load_study(
        study_name="xgb-diabetes",
        storage="sqlite:///models/optuna.db",
    )
    trials_df = study.trials_dataframe()

    fig = px.line(
        trials_df,
        x="number",
        y="value",
        title="Optuna Trial History (weighted F1)",
        labels={"number": "Trial", "value": "Weighted F1"},
        markers=True,
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    display_cols = ["number", "value"]
    for col in ["params_max_depth", "params_learning_rate"]:
        if col in trials_df.columns:
            display_cols.append(col)

    st.dataframe(trials_df[display_cols].head(10), use_container_width=True)
