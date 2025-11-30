.PHONY: install install-dev lint format check test clean help

# Main CI flow: Fails fast (Lint -> Type -> Test)
ci: lint type test

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Quality Checks
lint:
	ruff check src tests
	ruff check --select I src tests
	black --check src tests
	ruff format --check src tests

type:
	mypy src

# Testing (Generates coverage automatically via config)
test:
	pytest --cov --cov-report=xml --cov-report=term

# Formatting (Fixes code)
format:
	ruff check --select I --fix src tests
	black src tests
	ruff format src tests

# Cleaning (Cross-platform via Python)
clean:
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	python -c "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('*.egg-info')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.mypy_cache')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.ruff_cache')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('build')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('dist')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('htmlcov')]"
	@echo "Clean complete."

help:
	@echo "  install-dev  - Install dev dependencies & pre-commit hooks"
	@echo "  ci           - Run Linting -> Typing -> Tests (Use this for local check)"
	@echo "  format       - Auto-format code"
	@echo "  clean        - Remove all build/cache artifacts"
