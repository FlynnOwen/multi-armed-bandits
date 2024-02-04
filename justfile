#!/usr/bin/env just --justfile
PYENV_NAME := 'multi_armed_bandits'
PYTHON_VERSION := '3.11'

### Python Setup

# Complete installation/reinstallation of the repo pyenv
install-venv:
    pyenv virtualenv {{PYTHON_VERSION}} {{PYENV_NAME}}
    pyenv local {{PYENV_NAME}}

# Install requirements
install-requirements: install-venv
    pip3 install --upgrade pip
    pip3 install -r requirements.txt

# Activates repo pyenv if exists, else runs installation
pyenv-activate:
    pyenv local {{PYENV_NAME}} || just install-venv

# Testing
local-test: pyenv-activate
    python3 -m pytest

# Linting
lint:
    ruff check . --fix --exit-zero

format:
    ruff check . --fix
    ruff format .