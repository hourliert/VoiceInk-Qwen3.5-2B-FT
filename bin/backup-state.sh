#!/usr/bin/env bash
set -euo pipefail

readonly ROUTER_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly BACKUP_ROOT="${VOICEINK_BACKUP_ROOT:-${ROUTER_ROOT}/training/backups}"
readonly STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${BACKUP_ROOT}/annotations" "${BACKUP_ROOT}/mlflow"
"${ROUTER_ROOT}/.venv/bin/python3" "${ROUTER_ROOT}/src/data/manage.py" \
  backup >/dev/null

"${ROUTER_ROOT}/.venv/bin/python3" "${ROUTER_ROOT}/src/data/backup_mlflow.py" \
  --output-dir "${BACKUP_ROOT}/mlflow" --retain 30
echo "VoiceInk state backup completed at ${STAMP}"
