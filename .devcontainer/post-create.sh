#!/usr/bin/env bash
set -e

# Setup poetry and install
poetry config virtualenvs.in-project true
poetry install --with dev

# Setup and install pre-commit
cd /workspace
source .venv/bin/activate
pre-commit install