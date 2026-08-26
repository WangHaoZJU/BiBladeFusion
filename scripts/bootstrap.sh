#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
elite_wheel="${1:-/home/vale/Documents/Elite_Robots_CS_SDK_Python/dist/elite_cs_sdk-1.0.0-cp312-cp312-linux_x86_64.whl}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${project_dir}/.uv-cache}"

if [[ ! -f "${elite_wheel}" ]]; then
  echo "Elite CS SDK wheel not found: ${elite_wheel}" >&2
  exit 1
fi

cd "${project_dir}"
uv venv --python 3.12
uv sync --all-groups
uv pip install "${elite_wheel}"

uv run python -c "import elite_cs_sdk; print('elite_cs_sdk import: OK')"
uv run bbf version
