#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLFLOW_HOST="${MLFLOW_HOST:-192.168.1.150}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_ALLOWED_HOSTS="${MLFLOW_ALLOWED_HOSTS:-${MLFLOW_HOST}:${MLFLOW_PORT},localhost:${MLFLOW_PORT},127.0.0.1:${MLFLOW_PORT}}"
MLFLOW_CORS_ALLOWED_ORIGINS="${MLFLOW_CORS_ALLOWED_ORIGINS:-http://${MLFLOW_HOST}:${MLFLOW_PORT}}"

mkdir -p "${ROOT}/training/mlflow/artifacts"

exec "${ROOT}/.venv/bin/mlflow" server \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}" \
  --backend-store-uri "sqlite:///${ROOT}/training/mlflow/mlflow.db" \
  --artifacts-destination "${ROOT}/training/mlflow/artifacts" \
  --allowed-hosts "${MLFLOW_ALLOWED_HOSTS}" \
  --cors-allowed-origins "${MLFLOW_CORS_ALLOWED_ORIGINS}"
