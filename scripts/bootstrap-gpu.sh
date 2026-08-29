#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
elite_wheel="${1:-/home/vale/Documents/Elite_Robots_CS_SDK_Python/dist/elite_cs_sdk-1.0.0-cp312-cp312-linux_x86_64.whl}"
config_path="${2:-${project_dir}/configs/default.yaml}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${project_dir}/.uv-cache}"

if [[ ! -f "${elite_wheel}" ]]; then
  echo "Elite CS SDK wheel not found: ${elite_wheel}" >&2
  exit 1
fi
if [[ ! -f "${config_path}" ]]; then
  echo "BiBladeFusion configuration not found: ${config_path}" >&2
  exit 1
fi
if [[ ! -f "${project_dir}/third_party/FoundationStereo/core/foundation_stereo.py" ]]; then
  echo "FoundationStereo submodule is absent." >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; install and verify the NVIDIA driver first." >&2
  exit 1
fi

cd "${project_dir}"
nvidia-smi
uv venv --python 3.12
uv sync --locked --all-groups --all-extras

# The proprietary SDK is intentionally installed last because it is not declared in
# pyproject.toml and a later exact uv sync may otherwise remove it.
uv pip install --python "${project_dir}/.venv/bin/python" "${elite_wheel}"

"${project_dir}/.venv/bin/python" -c \
  "import elite_cs_sdk; print('elite_cs_sdk import: OK')"
"${project_dir}/.venv/bin/python" -c \
  "import torch; print('torch:', torch.__version__, 'runtime CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available()); assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"
"${project_dir}/.venv/bin/bbf" version
"${project_dir}/.venv/bin/bbf" stereo doctor --config "${config_path}"

echo "GPU runtime bootstrap completed. A real infer-session smoke test is still required."
