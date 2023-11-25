#!/usr/bin/env just --justfile
export PYTHON_VERSION := "3.10"

# Virtual environment
install-venv:
    [ -d venv ] || python{{PYTHON_VERSION}} -m venv venv

install-requirements: install-venv
    ./venv/bin/python -m pip install -r requirements.txt

# Testing
local-test: install-requirements
    ./venv/bin/python -m pytest

# Linting
lint:
    ruff .

format:
    black .