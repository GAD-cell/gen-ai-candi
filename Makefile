PYTHON_VERSION := 3.11
VENV := .venv
PYTHON := ./$(VENV)/bin/python3


TEST_MODEL_SCRIPT := src/test_model.py


.PHONY: env 

env:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "Setting up environment..."
	@rm -rf $(VENV)
	@uv venv $(VENV) --python $(PYTHON_VERSION)
	@echo "Installing dependencies..."
	@uv pip install -r requirements.txt --python $(PYTHON)
	@echo "Environment ready. Activate with: source $(VENV)/bin/activate"

test_model:
	@PYTHONPATH=. $(PYTHON) $(TEST_MODEL_SCRIPT)



clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV)
	@rm -rf __pycache__ src/__pycache__
	@echo "Cleanup complete"