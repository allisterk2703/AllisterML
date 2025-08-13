.PHONY: help install-requirements install-requirements-dev clean create-data-folder mlflow-ui black isort

SRC_DIR := src

help:  ## Displays the list of available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "🔹 \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install-requirements:  ## Installs main dependencies
	@pip install --upgrade pip setuptools wheel
	@pip install -r requirements.txt
	@echo "✅ Main dependencies installed"

install-requirements-dev:  ## Installs development dependencies
	@pip install --upgrade pip setuptools wheel
	@pip install -r requirements-dev.txt
	@echo "✅ Development dependencies installed"

clean:  ## Removes temporary files
	@find . -name "__pycache__" -type d -exec rm -rf {} +
	@find . -name "*.pyc" -type f -delete
	@rm -rf .pytest_cache catboost_info .ruff_cache log.log
	@find . -type d -name "logs" -exec rm -rf {} +
	@find . -type d -name "mlruns" -exec rm -rf {} +
	@echo "🧹 Temporary files removed"

create-data-folder: ## Creates a data folder
	@read -p "Enter the name of the dataset: " DATASET_NAME; \
	mkdir -p data/$${DATASET_NAME}; \
	mkdir -p data/$${DATASET_NAME}/{logs,mlruns}; \
	touch data/$${DATASET_NAME}/{config.yaml,data.csv,source.txt}; \
	touch data/$${DATASET_NAME}/{specific_transformations.py,feature_creation_and_transformation.py}; \
	echo "✅ Folder '$${DATASET_NAME}/' created in 'data/'"

mlflow-ui:  ## Launches MLflow UI for a specific dataset
	@DATASET_NAME=$$(ls -d data/*/ | xargs -n 1 basename | fzf --prompt="Choose a dataset: " --height=10 --reverse); \
	echo "🔍 Launching MLflow for dataset: $${DATASET_NAME}"; \
	python -m mlflow ui --backend-store-uri file://$(PWD)/data/$${DATASET_NAME}/mlruns --port 5001

isort:  ## Sort Python imports
	@echo "👷Sorting imports with isort..."
	@isort $(SRC_DIR)
	@echo "✅ Imports sorted with isort\n"

black:  ## Format Python code with Black
	@echo "🎨Formatting code with Black..."
	@black $(SRC_DIR)
	@echo "✅ Code formatted with Black\n"

ruff:  ## Check and fix Python code with Ruff
	@echo "👷Checking and fixing code with Ruff..."
	@ruff check $(SRC_DIR) --fix
	@ruff format $(SRC_DIR)
	@echo "✅ Code checked and fixed with Ruff\n"

pre-commit: isort black # ruff  ## Run all pre-commit checks without Git
	@echo "✅ Pre-commit executed"
