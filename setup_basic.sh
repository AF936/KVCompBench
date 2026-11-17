#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
if [ ! -f benchmark/settings.txt ]; then cp benchmark/settings.example.txt benchmark/settings.txt; fi
scratch="$project_root/.venvs/linux-x86_64-basic-cpu/tmp"
mkdir -p "$scratch"
export TMPDIR="$scratch" TEMP="$scratch" TMP="$scratch" PYTHONDONTWRITEBYTECODE=1
export PIP_CACHE_DIR="$project_root/.venvs/linux-x86_64-basic-cpu/pip-cache"
python3 -m venv --copies .venvs/linux-x86_64-basic-cpu
python=.venvs/linux-x86_64-basic-cpu/bin/python
"$python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
"$python" -m pip install -r benchmark/requirements/core.txt
"$python" -m pip check
