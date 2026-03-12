#!/usr/bin/env bash
set -euo pipefail

readonly ROUTER_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly LLAMA_SERVER="${LLAMA_SERVER:-/home/thomas/llama.cpp/llama-server}"
readonly LLAMA_HOST=127.0.0.1
readonly LLAMA_PORT=8002
readonly PROXY_HOST=0.0.0.0
readonly PROXY_PORT=8001
readonly PROXY_LOG_FILE="${ROUTER_ROOT}/logs/voiceink_proxy_requests.jsonl"

mkdir -p "${ROUTER_ROOT}/logs"

cleanup() {
  if [[ -n "${proxy_pid:-}" ]]; then
    kill "${proxy_pid}" 2>/dev/null || true
  fi
  if [[ -n "${llama_pid:-}" ]]; then
    kill "${llama_pid}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

"${LLAMA_SERVER}" \
  --host "${LLAMA_HOST}" \
  --port "${LLAMA_PORT}" \
  --parallel 1 \
  --ctx-size 16384 \
  --flash-attn on \
  --jinja \
  --metrics \
  --models-preset "${ROUTER_ROOT}/config/models.ini" \
  --models-max 1 &
llama_pid=$!

python3 "${ROUTER_ROOT}/src/voiceink_proxy/server.py" \
  --listen-host "${PROXY_HOST}" \
  --listen-port "${PROXY_PORT}" \
  --backend-host "${LLAMA_HOST}" \
  --backend-port "${LLAMA_PORT}" \
  --log-file "${PROXY_LOG_FILE}" &
proxy_pid=$!

wait -n "${llama_pid}" "${proxy_pid}"
