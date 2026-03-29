.PHONY: install train test lint serve streamlit optuna-ui gradio audit docker-build

install:
	pip install -r requirements.txt -r requirements-dev.txt

train:
	python -m src.training.train --config config/config.yaml

test:
	python -m pytest tests/ -v --tb=short --cov=src --cov-fail-under=70

lint:
	black src/ tests/ && flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203 && isort src/ tests/ && mypy src/ && bandit -r src/ -ll -ii

serve:
	python -m uvicorn src.api.app:app --reload --port 8000

streamlit:
	python -m streamlit run src/api/streamlit_app.py

optuna-ui:
	python -m optuna_dashboard sqlite:///models/optuna.db --host 0.0.0.0 --port 8080

gradio:
	python -m src.api.gradio_demo

audit:
	pip-audit -r requirements.txt

docker-build:
	docker build -t b2-xgboost-shap .
