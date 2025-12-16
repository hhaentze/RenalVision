.PHONY: install install-dev lint format check test clean help

# Main CI flow: Fails fast (Lint -> Type -> Test)
ci: lint type test

# Installation
define INSTALL_RADIOMICS_CORE
	@echo "--------------------------------------------------"
	@echo "🔧 Pre-installing PyRadiomics Build Dependencies..."
	@echo "--------------------------------------------------"
	pip install "numpy>=1.26.0,<2.0.0" versioneer
	pip install pyradiomics --no-build-isolation
endef

install:
	$(INSTALL_RADIOMICS_CORE)
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	pip install -e .

install-dev:
	$(INSTALL_RADIOMICS_CORE)
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[dev]"
	pre-commit install

# Quality Checks
lint:
	ruff check src tests
	ruff format --check src tests

type:
	mypy src

# Testing (Generates coverage automatically via config)
test:
	pytest --cov --cov-report=xml --cov-report=term

# Formatting (Fixes code)
format:
	ruff check --select I,F401 --fix src tests
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
