#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src"
python -c "import sys; print(sys.version)"
python scripts/smoke_import.py
ruff check .
mypy src || true
pytest -q
echo "All dev checks completed."
