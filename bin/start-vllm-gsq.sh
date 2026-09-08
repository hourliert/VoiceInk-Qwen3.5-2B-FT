#!/usr/bin/env bash
set -euo pipefail

readonly ROUTER_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly VLLM_BIN="${VLLM_BIN:-${ROUTER_ROOT}/.venv-vllm/bin/vllm}"
readonly VLLM_PYTHON="${VLLM_PYTHON:-${ROUTER_ROOT}/.venv-vllm/bin/python}"
readonly VLLM_MODEL="${VLLM_MODEL:-${ROUTER_ROOT}/models/Qwen3.8-27B-3Bit-GSQ}"
readonly VLLM_PATCH="${VLLM_MODEL}/patch_vllm_qwen35_embedding.py"
readonly VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
readonly VLLM_PORT="${VLLM_PORT:-8005}"
readonly VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-Qwen3.8-27B-3Bit-GSQ}"
readonly VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
readonly VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
readonly VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
readonly VLLM_CUDA_LIB="${ROUTER_ROOT}/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13/lib"

# Humming loads NVRTC dynamically. The CUDA 13 wheel keeps its builtins beside
# libnvrtc rather than in the host CUDA 12 ldconfig path. It also JIT-builds a
# small launcher with the ninja executable installed inside this environment.
export PATH="${ROUTER_ROOT}/.venv-vllm/bin:${PATH}"
export LD_LIBRARY_PATH="${VLLM_CUDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ ! -x "${VLLM_BIN}" || ! -x "${VLLM_PYTHON}" ]]; then
  echo "vLLM environment is missing; expected ${VLLM_BIN}" >&2
  exit 1
fi

if [[ ! -f "${VLLM_MODEL}/config.json" || ! -f "${VLLM_PATCH}" ]]; then
  echo "GSQ model or its required vLLM patch is missing from ${VLLM_MODEL}" >&2
  exit 1
fi

# The checkpoint quantizes its embeddings, which requires the model author's
# patch to vLLM 0.27.1. The patch is idempotent, so verify it on every launch.
"${VLLM_PYTHON}" "${VLLM_PATCH}"

exec "${VLLM_BIN}" serve "${VLLM_MODEL}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --served-model-name "${VLLM_SERVED_MODEL_NAME}" \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
