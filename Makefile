.PHONY: install install-dev lint format format-nbs test clean help

# Detect OS
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
else
    DETECTED_OS := $(shell uname -s)
endif

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Linting 
lint:
	ruff check src
	ruff check --select I src 
	black --check src
	ruff format --check src

# Formatting
format:
	ruff check --select I --fix src 
	black src
	ruff format src

format-nbs:
	ruff check --select I --fix notebooks
	black notebooks --include '\.ipynb$'
	ruff format notebooks

# Testing
test:
	mypy

# Cross-platform clean command
clean:
ifeq ($(DETECTED_OS),Windows)
	@echo "Cleaning build artifacts (Windows)..."
	@if exist build ( rmdir /s /q build && echo "  Removed build" ) else echo "  No build directory"
	@if exist dist ( rmdir /s /q dist && echo "  Removed dist" ) else echo "  No dist directory"
	@if exist .pytest_cache ( rmdir /s /q .pytest_cache && echo "  Removed .pytest_cache" ) else echo "  No .pytest_cache"
	@if exist .ruff_cache ( rmdir /s /q .ruff_cache && echo "  Removed .ruff_cache" ) else echo "  No .ruff_cache"
	@if exist .mypy_cache ( rmdir /s /q .mypy_cache && echo "  Removed .mypy_cache" ) else echo "  No .mypy_cache"
	@echo "Cleaning Python egg-info..."
	@for /d %%i in (*.egg-info) do @if exist "%%i" ( rmdir /s /q "%%i" && echo "  Removed %%i" )
	@echo "Cleaning Python caches..."
	@for /f "delims=" %%d in ('dir /s /b /ad __pycache__ 2^>nul') do @( rmdir /s /q "%%d" && echo "  Removed %%d" )
	@for /f "delims=" %%f in ('dir /s /b *.pyc 2^>nul') do @( del /q "%%f" )
	@for /f "delims=" %%f in ('dir /s /b *.pyo 2^>nul') do @( del /q "%%f" )
	@echo "Clean complete!"
else
	@echo "Cleaning build artifacts (Unix)..."
	@rm -rf build && echo "  Removed build" || echo "  No build directory"
	@rm -rf dist && echo "  Removed dist" || echo "  No dist directory"
	@rm -rf .pytest_cache && echo "  Removed .pytest_cache" || echo "  No .pytest_cache"
	@rm -rf .ruff_cache && echo "  Removed .ruff_cache" || echo "  No .ruff_cache"
	@rm -rf .mypy_cache && echo "  Removed .mypy_cache" || echo "  No .mypy_cache"
	@echo "Cleaning Python egg-info..."
	@rm -rf *.egg-info && echo "  Removed *.egg-info" || echo "  No egg-info directories"
	@rm -rf src/*.egg-info && echo "  Removed src/*.egg-info" || true
	@echo "Cleaning Python caches..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null && echo "  Removed __pycache__ directories" || echo "  No __pycache__ directories"
	@find . -type f -name '*.pyc' -delete 2>/dev/null && echo "  Removed *.pyc files" || true
	@find . -type f -name '*.pyo' -delete 2>/dev/null && echo "  Removed *.pyo files" || true
	@find . -type f -name '*.pyd' -delete 2>/dev/null && echo "  Removed *.pyd files" || true
	@echo "Clean complete!"
endif

# Help
help:
	@echo "Available commands:"
	@echo "  install        - Install package"
	@echo "  install-dev    - Install package with development dependencies"
	@echo "  lint           - Run all linting checks (read-only)"
	@echo "  format         - Format code with black and ruff"
	@echo "  format-nbs     - Format jupyter notebooks with black and ruff"
	@echo "  test           - Run tests"
	@echo "  clean          - Clean build artifacts and caches"
	@echo ""
	@echo "Detected OS: $(DETECTED_OS)"

# Default target
default: install-dev