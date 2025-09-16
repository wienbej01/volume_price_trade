#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src"
python -c "import sys; print(sys.version)"
python scripts/smoke_import.py
python -m ruff check --select E9,F63,F7,F82 .
python -m mypy src || true
python -m pytest -q
echo "All dev checks completed."
