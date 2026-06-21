#!/usr/bin/env bash
set -euo pipefail
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python scripts/check_environment.py
