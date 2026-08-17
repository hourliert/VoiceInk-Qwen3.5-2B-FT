#!/usr/bin/env bash
set -euo pipefail

readonly ROUTER_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly ASDF_SHIMS="${ASDF_SHIMS:-/home/thomas/.asdf/shims}"
readonly ASDF_BIN="${ASDF_BIN:-/home/linuxbrew/.linuxbrew/bin}"
export PATH="${ASDF_SHIMS}:${ASDF_BIN}:${PATH}"
readonly LLAMA_SERVER="${LLAMA_SERVER:-/home/thomas/llama.cpp/llama-server}"
readonly LLAMA_HOST=0.0.0.0
readonly LLAMA_PORT=8002
readonly PROXY_HOST=0.0.0.0
readonly PROXY_PORT=8001
readonly REVIEW_HOST="${REVIEW_HOST:-192.168.1.150}"
readonly REVIEW_PORT="${REVIEW_PORT:-8003}"
readonly MLFLOW_HOST="${MLFLOW_HOST:-192.168.1.150}"
readonly MLFLOW_PORT="${MLFLOW_PORT:-5000}"
readonly MLFLOW_AUTOSTART="${MLFLOW_AUTOSTART:-1}"
readonly PROXY_LOG_FILE="${ROUTER_ROOT}/logs/voiceink_proxy_requests.jsonl"

export MLFLOW_HOST MLFLOW_PORT

mkdir -p "${ROUTER_ROOT}/logs"

cleanup() {
  if [[ -n "${mlflow_pid:-}" ]]; then
    kill "${mlflow_pid}" 2>/dev/null || true
  fi
  if [[ -n "${review_pid:-}" ]]; then
    kill "${review_pid}" 2>/dev/null || true
  fi
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

"${ROUTER_ROOT}/.venv/bin/python3" "${ROUTER_ROOT}/src/labeling/live_review_server.py" \
  --host "${REVIEW_HOST}" \
  --port "${REVIEW_PORT}" \
  --log-file "${PROXY_LOG_FILE}" &
review_pid=$!

child_pids=("${llama_pid}" "${proxy_pid}" "${review_pid}")

if [[ "${MLFLOW_AUTOSTART}" == "1" ]]; then
  if curl --silent --fail --max-time 2 \
    "http://${MLFLOW_HOST}:${MLFLOW_PORT}/health" >/dev/null 2>&1; then
    echo "MLflow is already healthy at http://${MLFLOW_HOST}:${MLFLOW_PORT}; reusing it."
  else
    "${ROUTER_ROOT}/bin/start-mlflow.sh" &
    mlflow_pid=$!
    child_pids+=("${mlflow_pid}")
  fi
fi

wait -n "${child_pids[@]}"
