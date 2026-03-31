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

## Day 3 — 2026-03-30 — SHAP unified API, 4 plot types, calibration curve, Brier score
> Project: B2-XGBoost-SHAP

### What was done
- Created `src/evaluation/shap_analysis.py`: `compute_shap`, `_plot_beeswarm`, `_plot_global_bar`, `_plot_waterfalls`, `_plot_dependence`, `run_calibration`, `_update_results_json`, `main`.
- Discovered SHAP 0.49 unified API requires `model.predict_proba` callable, not the estimator directly; sliced `[:, :, 1]` for class-1 values.
- Generated all 4 SHAP artifact types: beeswarm PNG, Plotly HTML bar, waterfall PNGs (TP/TN/FP), dependence scatter PNGs.
- Calibrated XGBoost with `CalibratedClassifierCV(cv='prefit', method='isotonic')` on val set; logged Brier scores via MLflow.
- Added 14 tests in `tests/test_shap_analysis.py`; full suite 50/50, 90% coverage.

### Why it was done
- Raw model probabilities are often miscalibrated — Brier score quantifies this and calibration curve visualises it.
- SHAP explains individual predictions (waterfall) and global feature importance (beeswarm/bar) required for the portfolio dashboard.
- Feature importance JSON feeds the Streamlit app; Plotly HTML renders with `st.plotly_chart`.

### How it was done
- `shap.Explainer(model.predict_proba, X_train_df, feature_names=cols)` — DataFrame background preserves column names for `shap_values[:, "Glucose"]` slicing.
- `shap_values_full[:, :, 1]` slices the (n, features, 2) output to class-1 Explanation; all SHAP plot functions receive this.
- TP/TN/FP indices found with `np.where((y_test==1)&(y_pred==1))[0]`; fallback to index 0 if class absent.
- `matplotlib.use("Agg")` set before pyplot import; `# noqa: E402` suppresses flake8 E402 for all deferred imports.

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| shap.Explainer unified API | Auto-selects best explainer; consistent API across model types | shap.TreeExplainer directly | Fails in SHAP 0.49 with sklearn-wrapped XGBoost due to booster string parsing |
| CalibratedClassifierCV(cv='prefit') | Calibrates a pre-fitted model on held-out val set with no retraining | cv=5 (cross-val) | Would re-train the model — incompatible with Optuna-tuned weights |
| Isotonic regression calibration | Non-parametric; better than Platt scaling when sigmoid assumption fails | Platt (sigmoid) | Assumes sigmoid shape; less flexible for tree models |
| Plotly HTML for bar chart | Interactive, embeds in Streamlit with `st.plotly_chart` | Matplotlib static PNG | Static; no hover tooltips for feature values |

### Definitions (plain English)
- **Brier score**: Mean squared error between predicted probabilities and true binary labels; 0 = perfect, 0.25 = random, lower is better.
- **Calibration**: A model is calibrated if predicted probability 0.7 means ~70% of those cases actually belong to the positive class.
- **Waterfall plot**: A SHAP plot for one prediction showing how each feature pushed the output up or down from the baseline.
- **Dependence plot**: Scatter of SHAP value vs feature value for one feature; reveals non-linear effects and interactions.
- **predict_proba callable**: In SHAP 0.49+, passing `model.predict_proba` instead of `model` tells SHAP to treat the model as a black-box function rather than inspecting its internals.

### Real-world use case
- SHAP waterfall + calibration: Used at insurance companies (e.g. Allianz) to explain individual claim risk scores and ensure probability outputs are trustworthy for pricing.
- Brier score monitoring: Used in clinical decision support systems (Epic, Cerner) where a poorly calibrated probability could mislead a physician.
- Plotly SHAP bars: Used in Databricks AutoML to display feature importances in the interactive experiment UI.

### How to remember it
- **Calibration = a weather forecast analogy**: if your model says 70% chance of rain every day, but it only rains 40% of those days, it's *overconfident* and needs calibration.
- **Brier score = golf**: lower is better; 0 = hole in one, 0.25 = didn't even leave the tee.

### Status
- [x] Done
- Next step: Day 4 — Streamlit dashboard: load SHAP artifacts, render beeswarm/waterfall/dependence with `st.plotly_chart` and `st.image`.

---

## Day 4 — 2026-03-31 — Streamlit 4-tab interactive dashboard with Plotly SHAP
> Project: B2-XGBoost-SHAP

### What was done
- Built `src/api/streamlit_app.py` with four tabs: Predict, Global SHAP, Feature Dependence, HPO History.
- Tab 1: 8 sliders → impute → calibrated XGBoost predict → Plotly Waterfall SHAP per-sample.
- Tab 2: Plotly horizontal bar of mean |SHAP| + XGBoost vs LightGBM metrics table.
- Tab 3: Plotly scatter dependence plot (feature vs SHAP value, coloured by Glucose).
- Tab 4: Optuna study loaded from `models/optuna.db`; trial convergence as Plotly line chart.

### Why it was done
- Days 0–3 produced static reports; Day 4 wraps everything in a live interactive UI so stakeholders can explore predictions without code.

### How it was done
- `@st.cache_data` on `load_all()` so models/SHAP arrays are read once and reused across interactions.
- `st.tabs()` splits concerns cleanly; each tab is an independent `with` block.
- `explainer(input_imp)` (callable SHAP API) computes single-row SHAP values on demand.
- `go.Waterfall` renders per-prediction SHAP as a horizontal waterfall via Plotly (not `st.pyplot`).

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| Streamlit | Zero-boilerplate Python web UI, ideal for ML demos | Gradio | Less flexible multi-tab layout |
| Plotly | Interactive zoom/hover, embeds in Streamlit natively | Matplotlib/SHAP `.plot()` | Static images, poor mobile UX |
| `@st.cache_data` | Prevents re-loading 50 MB artifacts on every widget interaction | No caching | ~3s re-load on each slider move |
| `optuna.load_study` | Re-uses existing SQLite study without re-running HPO | Re-running trials | Wasteful and slow |

### Definitions (plain English)
- **Waterfall chart**: A bar chart that starts from a baseline and shows how each feature pushes the prediction up (red) or down (blue), ending at the final predicted probability.
- **`@st.cache_data`**: A Streamlit decorator that memoises a function's return value so it only executes once per session, not on every widget re-render.
- **Dependence plot**: A scatter plot of one feature's raw value (x-axis) vs its SHAP value (y-axis), revealing non-linear effects and interactions.
- **`optuna.load_study`**: Re-opens an existing Optuna experiment from its SQLite DB so you can query trial history without re-running HPO.

### Real-world use case
- Streamlit dashboards are used at Airbnb (price prediction explainer), Uber (model monitoring UI), and many Kaggle-winning teams to share ML findings with non-technical stakeholders.

### How to remember it
- Think of the app as four TV channels on one remote: Predict (live demo), Global (big picture), Dependence (zoom on one feature), HPO (behind-the-scenes tuning history).

### Status
- [x] Done
- Next step: Day 5 — FastAPI inference endpoint + Docker image + CI integration test.

---

## Day 5 — 2026-03-31 — FastAPI /predict with SHAP, 3-model evaluation, Gradio, Docker
> Project: B2-XGBoost-SHAP

### What was done
- Built `src/evaluation/evaluate.py`: loads 3 models (xgb, lgbm, calibrated_xgb), computes accuracy/F1/AUC/Brier, saves confusion-matrix PNGs and a 3-curve ROC figure, merges into `reports/results.json`, logs to MLflow.
- Built `src/api/app.py`: production FastAPI with slowapi rate-limit, CORSMiddleware, TrustedHostMiddleware, Content-Length guard, Prometheus /metrics, `GET /health`, `POST /predict` (SHAP), `GET /model_info`.
- Built `src/api/gradio_demo.py`: 8 sliders (bounds from train.csv), POST to localhost:8000, displays label + top-3 SHAP features.
- Built `hf_space/app.py`: identical UI hitting a remote API_URL env var; only gradio + httpx deps.
- Added multi-stage Dockerfile (builder pip install → runtime copy; non-root appuser; EXPOSE 8000).
- Added 16 new tests across `test_api.py` and `test_evaluate.py`; full suite 66/66 green, 75.8% coverage.

### Why it was done
- Need a REST endpoint so downstream apps can score patients without loading the model directly.
- Docker image enables reproducible deployment to any cloud provider.
- Gradio/HF Space gives a public interactive demo for the portfolio.

### How it was done
- **FastAPI lifespan** (`@asynccontextmanager`) loads `calibrated_xgb.joblib`, `imputer.joblib`, `shap_explainer.joblib` once into `_state` dict; every request reads from that dict.
- **SHAP shape handling**: `ExactExplainer` on a `CalibratedClassifierCV` returns shape `(1, n_features, n_classes)`; sliced as `values[0, :, 1]` to get class-1 float values for the JSON response.
- **Test isolation**: `ExitStack` patches `joblib.load` (side-effect=[model, imputer, explainer]) and `Path.exists` (returns True only for `*.joblib` paths) so the TestClient lifespan runs without touching disk.
- **Dockerfile**: builder stage pip-installs to system site-packages; runtime stage copies `/usr/local/lib/python3.12/site-packages` and only the necessary app folders.

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| FastAPI lifespan | Loads models once at startup; async-native; replaces deprecated `@on_event` | `@app.on_event("startup")` | Deprecated in FastAPI 0.93+ |
| slowapi | Drop-in rate limiter for FastAPI using Redis or in-memory; 1-line decorator | custom middleware | More code, no per-route granularity |
| prometheus-fastapi-instrumentator | Auto-instruments all routes; one `.instrument(app).expose(app)` call | manual Prometheus counters | Far more boilerplate |
| ExitStack (tests) | Composes multiple context-manager patches cleanly for module-scoped fixture | nested `with patch` | Deeply nested, hard to read |
| Multi-stage Dockerfile | Smaller final image: build tools stay in builder, not runtime | single-stage | Larger image, wider attack surface |

### Definitions (plain English)
- **Lifespan context**: an async generator that FastAPI calls on startup (before `yield`) and shutdown (after `yield`) — the right place to load heavy resources like ML models.
- **slowapi**: a rate-limiting library that wraps `limits` and integrates with FastAPI's `Request` object to count calls per IP per time window.
- **ExactExplainer**: SHAP's model-agnostic explainer that enumerates feature subsets exactly; used automatically for calibrated wrappers that TreeExplainer can't handle directly.
- **Multi-stage Docker build**: uses two `FROM` instructions so dependency build tools are discarded before the final image is created, reducing size and attack surface.

### Real-world use case
- FastAPI + slowapi rate-limiting is used by companies like Hugging Face to protect their Inference API endpoints from abuse without a separate API gateway.
- Multi-stage Docker builds are standard practice at Google, Netflix, and Stripe for shipping minimal production containers.

### How to remember it
- **lifespan = hotel check-in/check-out**: the lobby (startup) prepares your room (loads models); you use it during your stay (requests); checkout (shutdown) cleans up.
- **multi-stage Docker = build in a factory, ship in a box**: the factory has all the tools (compiler, pip), but only the finished product goes into the shipping box (runtime image).

### Status
- [x] Done
- Next step: Day 6 — Streamlit tab for live API calls + full CI Docker build test + README.

---

## Day 6 — 2026-03-31 — Tests 76% coverage, all 8 CI lint gates green, README added
> Project: B2-XGBoost-SHAP

### What was done
- Created `tests/test_model.py` with 3 artifact smoke tests (xgb_model_loads, imputer_no_nan, shap_values_shape).
- Fixed mypy (12 → 0 errors): installed `types-PyYAML`, added `# type: ignore[arg-type]` on slowapi handler and feature_engine MeanMedianImputer, renamed `col` → `param_col` in streamlit loop to fix DeltaGenerator/str reuse.
- Fixed flake8 E501 in `app.py` by splitting `add_exception_handler` call across lines.
- Wrote `README.md`: architecture diagram, benchmark table (3 models × 4 metrics), SHAP beeswarm screenshot, API usage, CI gate summary, paper references.
- All 8 CI gates passed: black+isort+flake8+mypy+bandit+radon+interrogate(81.5%)+pytest(76%)+pip-audit.

### Why it was done
- CI must be first-time green on GitHub Actions push — every lint gate must pass locally first.
- `tests/test_model.py` adds integration-level confidence that serialised artifacts are valid.
- README is required for the portfolio repo to communicate scope and benchmarks to reviewers.

### How it was done
- Ran each of the 8 CI steps in order, fixed failures before proceeding to the next.
- mypy errors traced to: invariant `list[str]` vs `list[str|int]` (feature_engine), slowapi handler signature narrowing (starlette), and variable shadowing (streamlit).
- `# type: ignore[arg-type]` used at the call site rather than disabling mypy globally.
- README written from `reports/results.json` for real benchmark numbers.

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| mypy | Static type checking catches runtime bugs before CI | pylance (IDE only) | Doesn't run in CI pipelines |
| types-PyYAML | Provides type stubs for yaml module | inline `# type: ignore` on every yaml call | Noisier and harder to maintain |
| `# type: ignore[arg-type]` | Silences specific mypy error at the call site | Disabling mypy for the whole file | Too broad — hides real bugs |
| bandit `-ll -ii` | Filters to medium+ severity and confidence only | Running without flags | Too many false positives on low-severity findings |

### Definitions (plain English)
- **type stub (.pyi file)**: A file containing only type annotations (no runtime code) that mypy reads to know the types of a library that wasn't originally typed.
- **invariant**: A type parameter where `list[str]` is NOT considered a subtype of `list[str|int]` — mutations could break type safety.
- **Brier score**: Mean squared error between predicted probability and actual binary outcome; 0 = perfect calibration, 0.25 = random.
- **cyclomatic complexity**: Count of independent paths through a function; high values indicate hard-to-test logic.

### Real-world use case
- Google, Stripe, and Meta run mypy as a required CI gate on all Python services — a PR cannot merge if mypy reports errors.
- SHAP-explained medical risk models (like this one) are required by several EU AI Act provisions to provide human-readable explanations for automated decisions.

### How to remember it
- CI pre-flight = "black formats → flake8 checks → mypy types → bandit secures → radon simplifies → interrogate documents → pytest tests → pip-audit audits". Run them in that exact funnel — each step catches a different class of defect.

### Status
- [x] Done
- Next step: Set GitHub repo topics, confirm Actions green on first push, deploy Gradio to HuggingFace Spaces.

---
