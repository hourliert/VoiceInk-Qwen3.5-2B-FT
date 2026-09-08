# VoiceInk Control Plane

The VoiceInk Control Plane is the LAN-only operating application at
http://192.168.1.150:8003. It is the everyday entry point for reviewing
production mistakes, completing canonical cohorts, sealing datasets, locating
training and evaluation runs, and checking the local stack.

MLflow remains the experiment system of record. The control plane shows concise
run summaries and deep-links into MLflow instead of duplicating its charts,
parameters, artifacts, prompts, and model lineage.

## Architecture

The production application has four explicit state boundaries:

    VoiceInk -> proxy:8001 -> llama-server:8002
                    |
                    +-> append-only proxy JSONL
                              |
                              v
                     incremental ingester
                              |
                              v
                      SQLite registry
                        ^     ^     ^
                        |     |     |
                    HTTP API Luna  durable maintenance
                        |
                        v
                 React + TypeScript UI:8003

SQLite is authoritative for private samples, immutable label revisions,
analyses, decisions, jobs, cohorts, releases, and lineage. Every thread owns its
own SQLite connection; WAL mode and bounded write transactions allow HTTP reads,
ten Luna workers, ingestion, and maintenance to coexist.

The ingester stores source inode, byte offset, and line number in SQLite. It
never acknowledges an incomplete JSONL line. Delivery is at least once and
sample upserts make replay safe. Rotation, truncation, process restart, and
partial final lines are covered by tests.

A decision commit writes only the immutable annotation/decision and a durable
maintenance outbox row. The HTTP response does not wait for JSONL export, audit
mirroring, or a 120 MB backup. A background worker coalesces pending rows,
regenerates those derived artifacts, retries transient failures three times,
and recovers running work after restart.

## Frontend behavior

The frontend lives under ui/ and is built with React, TypeScript, Vite, React
Router, and TanStack Query. The URL is the source of truth for routes and
collection filters. Server state is cached and cancellable; local editor state
belongs to the current sample. A delayed response from an old page cannot
replace the active route.

There is no page-wide refresh timer. Only a sample with a pending or running
Luna job polls its own detail endpoint every two seconds. After the evaluator
finishes, the proposal and independent evaluation appear together in place;
expanded session context, text selection, dropdowns, and unsaved edits remain
stable during that update. Failed jobs stop polling and expose an explicit Retry
action.

Production routes:

    /                                  overview and current blockers
    /review/recent                     latest proxy requests
    /review/queue                      canonical human-review queue
    /review/history                    searchable annotation registry
    /samples/{request_id}              durable sample review URL

    /data/cohorts                      canonical cohort list
    /data/cohorts/{name}               readiness and release action
    /data/releases                     sealed immutable releases
    /data/releases/{name}              manifest and lineage

    /models                            canonical, experimental, legacy inventory
    /models/{name}                     model details and promotions
    /runs/training                     concise MLflow training index
    /runs/evaluations                  concise MLflow evaluation index
    /runs/{mlflow_run_id}              run detail and direct MLflow link

    /system                            services, workers, ingestion, GPU
    /system/docs                       canonical and daily-review runbooks

Collection filters remain query parameters, for example:

    /review/queue?cohort=voiceink-bootstrap-v1&review=required&split=acceptance

## Daily labeling

Open Recent after noticing a production mistake. Opening a sample idempotently
queues Luna if no current analysis exists. The page presents:

1. raw transcript;
2. production output with readable edits against raw;
3. Luna proposal with readable edits against production;
4. independent validation, blind preference, score dimensions, confidence, and
   materiality;
5. one persistent final-label editor and session-context disclosure.

Choose Accept Luna, Save my edit, Production was correct, or Exclude. The
decision is committed before navigation. When reviewing a cohort, a successful
save advances only to the next sample that is Luna-ready, still pending, and
requires a human under the canonical policy. When no such sample remains, the
application returns to the filtered queue.

Fresh human decisions always override automatic policy. Automatic policy never
rewrites or deletes a human annotation.

## Cohort, release, training, and evaluation

The cohort page reports exact Luna coverage, required-human completion,
unresolved critical cases, split counts, and audit expansion. Release creation
is disabled in the UI until ready and the backend repeats the same fail-closed
check.

A release is immutable and manifest-addressed. Training is launched through the
canonical profile entry point:

    .venv/bin/python3 src/training/train.py sft       --profile qwen38-2b-sft       --release-manifest datasets/releases/RELEASE/manifest.json       --version VERSION       --export-gguf q4_k_m q8_0

The Training page monitors the concise MLflow run and links to the full MLflow
view. Promotion evaluation must complete both the locked 440 regression set and
the release's 150-sample acceptance split; see CANONICAL_PIPELINE.md.

## API and performance boundaries

The React app uses /api/v1. Key endpoints:

    GET  /api/v1/health
    GET  /api/v1/readiness
    GET  /api/v1/overview
    GET  /api/v1/samples
    GET  /api/v1/samples/{request_id}
    POST /api/v1/samples/{request_id}/analysis-jobs
    POST /api/v1/samples/{request_id}/decisions
    GET  /api/v1/cohorts/{name}/queue
    POST /api/v1/cohorts/{name}/analysis-jobs
    POST /api/v1/cohorts/{name}/releases
    GET  /api/v1/runs?category=training
    GET  /api/v1/runs/{run_id}

HTTP GET handlers never scan the proxy log, import records, update cohorts,
refresh exports, create backups, or run SQLite integrity checks. Cohort status
is a set-based, side-effect-free projection. Run collections return concise
metrics; complete metrics, parameters, and tags load only on a run detail page.

Measured through the production-shaped HTTP server on the reconciled registry:

- application shell: 0.6 ms p95;
- Overview: 4.9 ms p95;
- 50-row required-review queue: 37.5 ms p95;
- sample detail: 1.5 ms p95;
- durable decision commit: 17.3 ms, with maintenance completed asynchronously;
- full 148 MB first reconciliation: about 10 seconds in the background;
- zero-lag restart check: about 2 ms.

## Installation, build, and startup

Install the pinned frontend dependencies once:

    npm --prefix ui ci

The normal startup remains one command:

    bin/start.sh

It builds the local Vite production bundle, then supervises llama-server,
the VoiceInk proxy, the port-8003 control plane, and MLflow. The systemd unit
invokes this same script.

Frontend development and tests:

    npm --prefix ui run dev
    npm --prefix ui run build
    npm --prefix ui run test
    npm --prefix ui run test:e2e

Backend tests:

    .venv/bin/python3 -m unittest discover -s tests -p 'test_*.py'

## Health and recovery

Cheap liveness and readiness:

    curl http://192.168.1.150:8003/api/v1/health
    curl http://192.168.1.150:8003/api/v1/readiness

Health reports Luna worker count, queue depth, ingestion cursor/lag, and durable
maintenance state. Readiness performs only a minimal database read plus cursor
state. Full SQLite integrity belongs to backup/verification workflows, not
request handlers.

Create an online registry and MLflow backup:

    bin/backup-state.sh

The pre-migration backup and post-reconciliation backup remain under
datasets/registry/backups. To validate a restore, copy a backup to a disposable
path, open it with AnnotationStore, run quick_check, and reconcile the recorded
table counts before replacing any live file. Never overwrite the live registry
while the service is running.

## Legacy isolation

The former inline global-DOM frontend is preserved only under
src/control_plane/legacy. Production imports no code from that directory.
Historical datasets, runs, model binaries, and evaluation outputs remain in
their explicit legacy namespaces; the control plane does not scan them into the
canonical release hierarchy.
