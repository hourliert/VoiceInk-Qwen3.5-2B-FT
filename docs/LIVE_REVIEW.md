# VoiceInk Review Workflow

The Review section of the VoiceInk Control Plane turns recent production mistakes into durable,
human-approved training labels while the surrounding context is still fresh.
It is available at `http://192.168.1.150:8003` and is started automatically by
`bin/start.sh` alongside llama-server and the VoiceInk proxy.

The site has no authentication. It exposes private transcripts, clipboard text,
and screen OCR to devices that can reach the LAN address.

## Daily workflow

The Recent page shows the ten latest successful proxy requests. Opening a row
queues two independent ephemeral Codex CLI calls using `gpt-5.6-luna` at
`xhigh` reasoning effort by default:

1. the labeler creates the proposed gold cleanup;
2. the reference-free evaluator validates that proposal and blindly compares it
   with the production response using the strict VoiceInk dimensions.

Analysis is cached by request, response, model, reasoning effort, and prompt
hashes. Reopening a row does not spend another call unless its analysis failed
and Retry is selected.

Choose one final outcome:

- **Accept Luna** approves the proposed label;
- **Save My Edit** stores an immutable human revision;
- **Production Was Correct** approves the production response as the label;
- **Exclude** records the review but prevents training use.

Every decision commits immediately to SQLite and schedules a durable maintenance
outbox job. A background worker refreshes the approved JSONL export, mirrors the
append-only audit log, and makes an online SQLite backup without delaying the UI.

## Durable data

Private generated data is ignored by Git:

- `datasets/registry/annotations.sqlite3` — canonical transactional registry;
- `datasets/registry/events.jsonl` — append-only audit mirror;
- `datasets/registry/backups/` — rotating local SQLite backups;
- `datasets/exports/live/current-approved.jsonl` — current training-compatible export;
- `datasets/exports/live/current-approved-manifest.json` — counts and hashes.

The registry keeps samples, immutable annotation revisions, Luna analyses,
human decisions, job attempts, source-file provenance, prompt/model metadata,
and dataset-release membership. MLflow receives prompt versions, hashes,
scores, timings, and outcomes, but never raw transcripts or labels.

Check health:

```bash
curl http://192.168.1.150:8003/api/health
```

## Historical migration and exports

The migration is idempotent and never changes, moves, or deletes legacy files:

```bash
.venv/bin/python3 src/labeling/migrate_annotations.py --dry-run
.venv/bin/python3 src/labeling/migrate_annotations.py
```

Refresh the current locked-eval-safe JSONL export:

```bash
.venv/bin/python3 src/labeling/export_annotations.py
```

Freeze an immutable release for a future training run:

```bash
.venv/bin/python3 src/labeling/export_annotations.py \
  --release-name voiceink-live-v1 \
  --release-dir datasets/exports/releases/voiceink-live-v1
```

The export is compatible with the existing preparation interface:

```bash
.venv/bin/python3 src/training/prepare_dataset.py \
  --input datasets/exports/releases/voiceink-live-v1/approved.jsonl
```

Release creation excludes the locked 440 samples using normalized transcript
fingerprints and preserves registry provenance for every exported revision. It
never starts a training run automatically.

## Manual launch

For an isolated debugging instance, build the React bundle and use a non-production port:

```bash
npm --prefix ui run build
.venv/bin/python3 src/control_plane/server.py \
  --host 127.0.0.1 --port 18003 --no-mlflow
```

The former `live_review_server.py` UI is legacy worker infrastructure, not the
production website. In normal operation, start or restart
`llama-router.service`; `bin/start.sh` supervises llama-server, the proxy,
MLflow, and this control plane as one stack.
