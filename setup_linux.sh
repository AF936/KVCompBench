#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_command="${PYTHON:-python3}"
if ! command -v "$python_command" >/dev/null 2>&1; then
    echo 'Install 64-bit Python 3.10+ with venv support, then rerun this script.' >&2
    exit 1
fi
exec "$python_command" "$project_root/benchmark/compression_framework/setup_environment.py" "$@"
