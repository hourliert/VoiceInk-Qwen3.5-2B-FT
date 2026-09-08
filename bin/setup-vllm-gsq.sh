#!/usr/bin/env bash
set -euo pipefail

readonly ROUTER_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly MODEL_REPO="${MODEL_REPO:-ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ}"
readonly MODEL_DIR="${MODEL_DIR:-${ROUTER_ROOT}/models/Qwen3.8-27B-3Bit-GSQ}"
readonly VLLM_ENV="${VLLM_ENV:-${ROUTER_ROOT}/.venv-vllm}"
readonly PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.12}"

for command in hf uv; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Required command is missing: ${command}" >&2
    exit 1
  fi
done

hf download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"

if [[ ! -x "${VLLM_ENV}/bin/python" ]]; then
  uv venv --python "${PYTHON_BIN}" "${VLLM_ENV}"
fi

uv pip install --python "${VLLM_ENV}/bin/python" vllm==0.27.1
"${VLLM_ENV}/bin/python" "${MODEL_DIR}/patch_vllm_qwen35_embedding.py"

echo "Qwen3.8 GSQ is installed. Start it with bin/start-vllm-gsq.sh."
