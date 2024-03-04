#!/usr/bin/env just --justfile
PYENV_NAME := 'multi_armed_bandits'
PYTHON_VERSION := '3.11'

### Python Setup

# Complete installation/reinstallation of the repo pyenv
install-venv:
    pyenv virtualenv {{PYTHON_VERSION}} {{PYENV_NAME}}
    pyenv local {{PYENV_NAME}}

# Activates repo pyenv if exists, else runs installation
pyenv-activate:
    pyenv local {{PYENV_NAME}} || just install-venv

# Install requirements
install-requirements: pyenv-activate
    pip3 install --upgrade pip
    pip3 install -r requirements.txt

# Testing
local-test: pyenv-activate
    python3 -m pytest

# Linting
lint:
    ruff check . --fix --exit-zero

format:
    just lint
    ruff format .

simulate-from-json COMMAND CONFIG: pyenv-activate
    python3 -m src.main simulate-from-json {{COMMAND}} {{CONFIG}}

# Usage
list-distributions:
    python3 -m src.main list-distributions

help *ARGS:
    python3 -m src.main {{ARGS}} --help

run COMMAND *ARGS:
    python3 -m src.main --{{COMMAND}} {{ARGS}}