# Research Notes — B2 XGBoost + SHAP
## Papers I Read Before Starting
- XGBoost (Chen & Guestrin, 2016) — gradient boosting with regularization
- SHAP (Lundberg & Lee, 2017, NeurIPS) — unified framework for feature attribution
- LightGBM (Ke et al., 2017) — leaf-wise growth, faster than XGBoost on large data
## Architecture Decisions
- XGBoost + LightGBM: model-agnostic explainability story, second results column
- Optuna over GridSearch: Bayesian 10x more efficient for 9 hyperparameters
- feature-engine imputation: zeros in glucose/BMI are biologically impossible
- Calibration curve: raw XGBoost probabilities are often poorly calibrated
- Plotly SHAP: interactive — far better for demos than static matplotlib
## What I Would Do With More Data
- Add SMOTE for class imbalance (diabetes=1 is 35% of dataset)
## Surprising Findings
- [Fill in after training]
