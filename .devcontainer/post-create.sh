#!/usr/bin/env bash
set -e

sudo chown vscode .venv || true

poetry config virtualenvs.in-project true

poetry install --with dev
