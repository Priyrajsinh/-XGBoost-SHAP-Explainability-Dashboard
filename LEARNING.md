# B2: XGBoost + SHAP Explainability Dashboard
# Owner: Priyrajsinh Parmar | github.com/Priyrajsinh
Stack: Python 3.12, xgboost, lightgbm, shap, optuna, optuna-dashboard,
       feature-engine, scikit-learn, pandas, numpy, streamlit, plotly,
       mlflow, fastapi, uvicorn, gradio, pandera, pydantic, slowapi,
       prometheus-fastapi-instrumentator, python-json-logger, bandit
Config: ALL hyperparameters in config/config.yaml — never hardcode
Logging: get_logger from src/logger.py
Exceptions: all errors through src/exceptions.py
Validation: pandera DIABETES_SCHEMA in src/data/validation.py before any split
Schemas: PredictInput / PredictOutput Pydantic in src/data/schemas.py
Security: bandit -r src/ -ll must return zero findings
Dataset: Pima Indians Diabetes (768 samples, 8 features, binary)
Imputation: feature-engine MeanMedianImputer on zero-as-NaN columns
Models: XGBoost + LightGBM (comparison), Optuna 50-trial Bayesian HPO
Explainability: SHAP (beeswarm, waterfall, dependence, force) via Plotly in Streamlit
Calibration: sklearn CalibrationDisplay + Brier score (required for medical data)
Registry: MLflow Model Registry — DiabetesClassifier in Staging
Docker image: b2-xgboost-shap (hardcoded everywhere)
NEVER commit: models/xgb_model.joblib models/lgbm_model.joblib
Commands: make install/train/test/lint/serve/streamlit/optuna-ui/gradio/audit/docker-build
