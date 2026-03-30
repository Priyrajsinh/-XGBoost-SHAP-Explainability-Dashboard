---

## Day 0 — 2026-03-29 — B2 scaffold: XGBoost + SHAP Explainability Dashboard
> Project: B2-XGBoost-SHAP

### What was done
- Created virtual environment (Python 3.10), upgraded pip to 26.0.1.
- Appended B2 sections to `~/.claude/skills/data-scientist.md` and `machine-learning-engineer.md` (preserved prior content).
- Created full project scaffold: `src/`, `config/`, `data/`, `models/`, `tests/`, `hf_space/`, `utils/`, `.github/`.
- Wrote `config/config.yaml` with XGBoost, LightGBM, Optuna, data, API, SHAP, MLflow and Docker config.
- Wrote `src/exceptions.py`, `src/logger.py`, `src/models/base.py`, `src/data/schemas.py`, `src/data/validation.py`.
- Created `Makefile`, `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `.gitignore`.
- Installed all packages from `requirements.txt` + `requirements-dev.txt`.
- Ran `bandit -r src/ -ll` → **No issues identified**, and `pytest tests/` → **3/3 passed**.

### Why it was done
- Bridge Project B2 needed a production-grade scaffold before any training or modelling begins.
- Skills files needed B2-specific SHAP, Optuna, and medical-data rules appended for reuse across future sessions.

### How it was done
- Used `pandera.DataFrameSchema` with strict=True to lock column names and types for the Pima dataset.
- Used Pydantic `BaseModel` + `@validator` for API request/response schemas with zero-glucose guard.
- Exception hierarchy inherits from `ProjectBaseError` so callers can catch broadly or specifically.
- All hyperparameters written only into `config/config.yaml`; source files contain no magic numbers.

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| XGBoost | Gold-standard gradient boosting, SHAP native support | sklearn GradientBoosting | 10× slower, no built-in SHAP tree explainer |
| SHAP | Unified API, TreeExplainer exact values for trees | LIME | Approximate, not consistent with Shapley axioms |
| Optuna | Bayesian HPO, SQLite storage, optuna-dashboard UI | GridSearchCV | Exponential search space; no resume-from-checkpoint |
| feature-engine | pandas-native imputer, pipeline compatible | sklearn SimpleImputer | Less readable; feature-engine has medical-zero handling |
| pandera | Schema validation with column-level checks | Great Expectations | Overkill for 9-column dataset; pandas-native pandera is simpler |

### Definitions (plain English)
- **SHAP value**: The amount each feature contributed to pushing a model's prediction away from the average prediction.
- **Optuna trial**: One candidate set of hyperparameters evaluated during Bayesian search; study = all 50 trials together.
- **Calibration**: Adjusting a model so that when it says "70% probability", the true frequency is also ~70%.
- **pandera schema**: A contract on a DataFrame's column names, dtypes, and value ranges enforced at runtime.
- **scale_pos_weight**: XGBoost parameter that upweights the minority class; set to (negatives/positives) ≈ 1.85 for Pima.

### Real-world use case
- SHAP + XGBoost: Used at Airbnb for pricing model explanations shown to hosts; at hospitals for ICU mortality risk scoring.
- Optuna: Used at Preferred Networks (creators) and DeepMind for large-scale NAS and RL hyperparameter search.
- MLflow Model Registry: Used at Databricks customers for A/B model promotion workflows.

### How to remember it
- **SHAP** = "How much did each player contribute to the team's win vs. average?" — Shapley values from cooperative game theory.
- **Optuna** = A smart guesser: it learns from past trials where good hyperparameters live, unlike GridSearch which guesses blindly.

### Status
- [x] Done
- Next step: Day 1 — download Pima dataset, run pandera validation, apply feature-engine imputation, create stratified splits.

---

## Day 1 — 2026-03-30 — Pima EDA, zero-as-NaN, imputer fitted on train, stratified splits
> Project: B2-XGBoost-SHAP

### What was done
- Implemented `src/data/dataset.py`: `load_pima()` downloads Pima CSV, replaces zeros with NaN for 5 columns, computes SHA-256 checksum, validates via pandera, saves `data/raw/pima.csv`.
- Implemented `src/data/preprocessing.py`: `impute_and_split()` does stratified 70/15/15 split FIRST, then fits `MeanMedianImputer` on train only, saves processed CSVs + `models/imputer.joblib`.
- Implemented `src/data/eda.py`: generates `class_dist.png`, `correlation.png`, `feature_distributions.png` in `reports/figures/`.
- Fixed `validation.py`: Glucose check changed from `gt(0)` to `ge(0)` so `fillna(0)` schema check works on raw data (API still enforces `> 0` separately).
- Added 27 tests across 3 new test files; CI gate all green, 98% coverage.

### Why it was done
- Raw Pima CSV has biologically impossible zeros (e.g. 374 rows with Insulin=0) — these must become NaN before any statistics are computed.
- Imputer must be fit on train-only to prevent data leakage from val/test statistics influencing train processing.
- EDA figures are required artefacts for the portfolio dashboard and explainability writeup.

### How it was done
- `StratifiedShuffleSplit` applied twice: first to carve out test (15%), then to carve val from the remaining 85% (rescaled fraction = 0.15/0.85).
- `MeanMedianImputer(imputation_method='median', variables=[...])` — median chosen because these medical features are right-skewed (Insulin median 32 vs mean ~80).
- SHA-256 computed over `df.to_csv(index=False).encode()` for a deterministic, column-order-stable hash.
- All paths use `pathlib.Path`; monkeypatching in tests redirects them to `tmp_path` — no real file I/O in tests.

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| feature-engine MeanMedianImputer | Pandas-native, sklearn-pipeline-compatible, variable-list API | sklearn SimpleImputer | Requires numpy arrays; column names lost without extra wrappers |
| StratifiedShuffleSplit | Preserves class ratio (65/35) in every split | train_test_split(stratify=) | Single-pass only; need two passes for three-way split |
| joblib.dump | Standard sklearn ecosystem serialiser, fast for numpy arrays | pickle | joblib handles large numpy arrays with mmap; safer for ML objects |
| hashlib.sha256 | Built-in, deterministic; detects raw-data drift across runs | MD5 | Collision-prone; SHA-256 is industry standard for data checksums |

### Definitions (plain English)
- **Data leakage**: Using information from the test set during training — e.g., fitting a scaler on all data means the model has "seen" test statistics.
- **Stratified split**: A split that keeps the same class ratio (e.g. 65% non-diabetic) in every subset, so no split is accidentally all one class.
- **Median imputation**: Replacing missing values with the middle value of the observed distribution; more robust than mean when outliers exist.
- **SHA-256 checksum**: A 64-character fingerprint of a file; if even one byte changes, the fingerprint changes completely.
- **zero-as-NaN**: Treating physiological zeros (e.g., BMI=0, Insulin=0) as missing data because they are biologically impossible.

### Real-world use case
- Data leakage prevention (fit-on-train-only): Standard practice in all production ML pipelines at Google, Meta, Netflix — a leaked scaler can cause a model that looks 95% accurate in test but fails in prod.
- SHA-256 data checksums: Used in DVC (Data Version Control) and MLflow to detect dataset drift between training runs.
- Stratified splits: Used whenever the positive-class rate is < 30%; common in medical AI (diabetic retinopathy, cancer detection).

### How to remember it
- **Fit on train, transform all**: Think of the imputer as a "recipe" — you write the recipe from tasting only the training soup, then use the same recipe on every bowl served later.
- **Two-pass stratified split**: Split the pie into slices in two cuts — first cut out 15% test, then cut 15/85 off the remainder for val.

### Status
- [x] Done
- Next step: Day 2 — XGBoost + LightGBM baseline training, MLflow experiment tracking, Optuna hyperparameter tuning.

---

## Day 2 — 2026-03-30 — Optuna HPO + XGBoost + LightGBM + MLflow Registry
> Project: B2-XGBoost-SHAP

### What was done
- Created `src/training/train.py` with `run_optuna_study()`, `train_final_models()`, `_register_to_mlflow()`, and `main()` CLI.
- 50-trial Optuna TPE study tunes 9 XGBoost hyperparameters; persists to SQLite for optuna-dashboard replay.
- Trained final XGBoost (Optuna best params) and LightGBM (config params); saved as `.joblib`.
- Logged best params, val F1, and model artifact to MLflow; transitioned registered model to Staging.
- Added `tests/test_train.py` with 9 tests; full suite 36/36 passing at 96% coverage.

### Why it was done
- Grid/random search is inefficient for 9 continuous hyperparameters; TPE converges in ~50 trials.
- Two model types (XGB + LGBM) enable direct comparison before choosing what to explain with SHAP.
- MLflow Model Registry enables versioned lifecycle management for staged rollouts (Staging → Production).

### How it was done
- `optuna.create_study` with `TPESampler(seed=42)` + `MedianPruner` + SQLite storage for reproducibility.
- `study.best_params` unpacked into `XGBClassifier`; LightGBM params read directly from `config["model"]["lgbm"]`.
- `mlflow.sklearn.log_model` + `mlflow.register_model` + `client.transition_model_version_stage("Staging")`.
- Tests use `monkeypatch` to redirect `_MODELS_DIR` / `_PROCESSED_DIR`; MLflow fully mocked with `unittest.mock.patch`.

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| Optuna TPE | Bayesian sequential search converges in ~50 trials | GridSearchCV | Exponential search space; no pruning |
| MedianPruner | Kills unpromising trials early, saves compute | No pruner | All 50 trials run to completion regardless |
| SQLite storage | Enables optuna-dashboard for live trial visualisation | In-memory | Lost on process exit; no dashboard support |
| mlflow.sklearn.log_model | Unified sklearn-compatible API for XGBoost | mlflow.xgboost.log_model | Less flexible when wrapping in sklearn Pipelines |
| joblib.dump | Fastest for tree-based numpy-heavy estimators | pickle | joblib ~2× faster for large numpy arrays |

### Definitions (plain English)
- **TPE Sampler**: Tree-structured Parzen Estimator — models which hyperparameter values scored well and samples more from those regions next.
- **MedianPruner**: Stops a trial early if its score is below the median of all completed trials at the same step.
- **MLflow Model Registry**: A versioned catalog of trained models with lifecycle stages: None → Staging → Production → Archived.
- **best_params**: The hyperparameter dictionary from the trial with the highest objective value in the Optuna study.

### Real-world use case
- Spotify uses Optuna for recommendation model tuning; DoorDash uses MLflow Model Registry for staged rollouts of demand forecasting models.

### How to remember it
- Optuna is a "smart random search": it remembers which hyperparameter regions scored well and explores there next — like a detective narrowing suspects after each clue.

### Status
- [x] Done
- Next step: Day 3 — SHAP explainability (shap.Explainer unified API, summary/waterfall/dependence/force plots).

---
