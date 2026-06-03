PYTHON := C:/Users/fikri/AppData/Local/Programs/Python/Python311/python.exe
PIP    := C:/Users/fikri/AppData/Local/Programs/Python/Python311/Scripts/pip.exe

.PHONY: install test analysis hte all lint clean

## Install all dependencies
install:
	$(PIP) install -r requirements.txt

## Run unit tests
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

## Run test with coverage
test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

## Run full analysis pipeline (frequentist + Bayesian + CUPED + sequential)
analysis:
	$(PYTHON) scripts/run_analysis.py

## Run HTE / uplift modeling (takes 5-15 min)
hte:
	$(PYTHON) scripts/run_hte.py

## Run everything end-to-end
all: analysis hte

## Lint source files
lint:
	$(PYTHON) -m flake8 src/ scripts/ app/ tests/ --max-line-length=110 --ignore=E501,W503

## Launch Gradio app locally
app:
	$(PYTHON) app/gradio_app.py

## View MLflow UI
mlflow-ui:
	$(PYTHON) -m mlflow ui

## Clean generated files
clean:
	rm -f data/processed/*.json
	rm -rf mlruns/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
