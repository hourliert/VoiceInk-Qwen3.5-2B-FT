# Canonical VoiceInk model lifecycle

The project now has one durable annotation registry, one immutable dataset
release format, one profile-driven training entry point, and one strict
promotion path. Private transcripts, releases, checkpoints, and binaries stay
local; MLflow receives native dataset fingerprints, counts, prompts, metrics,
lineage tags, and local output references.

## Data lifecycle

Create the first deterministic cohort:

```bash
.venv/bin/python3 src/data/manage.py bootstrap voiceink-bootstrap-v1
.venv/bin/python3 src/data/manage.py status voiceink-bootstrap-v1

# Queue fresh Luna analysis for the exact cohort (resumable/idempotent)
.venv/bin/python3 src/data/queue_luna.py voiceink-bootstrap-v1 \
  --server http://192.168.1.150:8003 --parallel 10
```

The cohort is exactly 1,200 unique samples selected with seed `20260817`: 900
train, 150 validation, and 150 sealed acceptance. The fixed strata are
representative (420), context-rich (240), technical/entity/number risk (240),
long (180), and no-op (120). The historical 440 is excluded and remains a
regression benchmark only. Synthetic examples are excluded from the bootstrap;
future releases may include at most 90, only in train.

Opening a bootstrap item in the control plane Review queue starts
fresh Luna labeling and independent evaluation. A release remains blocked until all 1,200 have a fresh Luna analysis, the
2% deterministic audit is complete, and genuinely ambiguous or unsafe proposal
cases are reviewed. High-confidence production wins keep the existing approved
label; high-confidence, validation-passing Luna wins are selected automatically
even when material. Passing ties and non-material, validation-passing comparisons also retain the existing approved cohort label. Fresh human decisions always take precedence.

After review, the preferred path is **Control plane → Data → Cohorts →
voiceink-bootstrap-v1**. The preview freezes all current human-reviewed
corrections made outside the cohort since cohort creation as an explicit
train-only delta. Validation (150), acceptance (150), and the locked 440 stay
unchanged. Enter a release name and seal it; creation fails if the selection
changed since the preview.

The CLI equivalent is:

```bash
.venv/bin/python3 src/data/manage.py release \
  voiceink-bootstrap-v1 voiceink-data-v1
.venv/bin/python3 src/data/manage.py verify \
  datasets/releases/voiceink-data-v1/manifest.json
```

Release directories are immutable and contain fingerprinted Qwen text-block and
LFM plain-text representations for each split, plus a signed `manifest.json`.
Both representations have identical sample IDs and lineage. Training never consumes
acceptance; checkpoint selection uses validation only. The manifest records the
correction cutoff, exact request and annotation IDs, expected split counts, and
selection SHA-256. The release page displays the manifest and copyable Qwen3.5
2B preflight/full-run commands.

## Training

All new jobs require a release manifest:

```bash
.venv/bin/python3 src/training/train.py sft \
  --profile qwen35-2b-sft \
  --release-manifest datasets/releases/voiceink-data-v1/manifest.json \
  --version v2 \
  --export-gguf q4_k_m q8_0
```

Use `--check-only` for a full trainer preflight or `--print-command` to inspect
the resolved low-level command. Profiles live in
`config/training_profiles.toml`. The low-level Qwen/LFM/DPO engines remain
implementation details so their memory-safe Unsloth behavior is shared; old
model-specific profile launchers are not canonical entry points.

Each MLflow training run records:

- native train and validation Dataset inputs with full SHA-256 tags;
- release name, release fingerprint, manifest path, profile, Git commit, and
  dirty-tree state;
- parameters, trainer metrics, terminal evaluation, system metrics, prompts,
  checkpoint/adapter/GGUF references, and an external Logged Model;
- no raw transcript rows and no copied private model binaries.

Dependencies are pinned by `pyproject.toml` and `uv.lock`. After the current
GPU job has finished, synchronize once with `uv sync`; `nvidia-ml-py` enables
MLflow GPU telemetry on future runs.

### Preference tuning

DPO uses the same fail-closed contract. First seal reviewed pairs as a
derivative of the exact SFT release; the command snapshots the Luna judgments
and judge prompt, rejects train/validation overlap and any acceptance leakage,
and fingerprints the parent manifest:

```bash
.venv/bin/python3 src/data/seal_preference.py voiceink-data-v1-dpo-v1 \
  --train datasets/preferences/train.jsonl \
  --validation datasets/preferences/validation.jsonl \
  --parent-manifest datasets/releases/voiceink-data-v1/manifest.json \
  --judgments datasets/preferences/judgments.jsonl \
  --judge-prompt src/eval/judge_prompt_strict_v2.codex.txt

.venv/bin/python3 src/training/train.py dpo \
  --profile qwen35-2b-dpo \
  --release-manifest datasets/releases/voiceink-data-v1-dpo-v1/manifest.json \
  --version v1 --export-gguf q4_k_m q8_0
```

The acceptance split belongs only to the parent SFT release and is never copied
into preference training.

## Evaluation and promotion

Start with the release's sealed 150-sample acceptance split:

```bash
.venv/bin/python3 src/eval/run.py \
  --baseline VoiceInk-Production \
  --candidate Qwen3.8-2B-VoiceInk-v2 \
  --release-manifest datasets/releases/voiceink-data-v1/manifest.json \
  --parallel 10
```

The release-aware entry point resolves the acceptance file, expected count,
strict-v3 rubric, and Codex/Luna provider. It does not run the historical 440.

Once a candidate is promising, add `--include-regression` to the same command
for the final promotion gate. That opt-in run adds the locked 440-sample
strict-v2 regression suite and produces complete promotion evidence. Any judge
failure makes the final command non-promotable. The promotion report enforces
protected-dimension significance, overall and dimension margins, and p50/p90
latency limits across both corpora, plus zero critical acceptance failures and
full release reproducibility.

Only a passing report can be promoted:

```bash
.venv/bin/python3 src/training/promote.py \
  --evidence results/promotion-evidence.json \
  --release-manifest datasets/releases/voiceink-data-v1/manifest.json \
  --model-path models/Qwen3.8-2B-VoiceInk-v2_gguf/Qwen3.8-2B.Q4_K_M.gguf \
  --model-name VoiceInk-Cleanup \
  --model-version qwen38-v2 \
  --mlflow-run-id RUN_ID
```

Promotion creates an MLflow registered-model version and `candidate` alias,
records the local decision, and updates `[VoiceInk-Production]`. It deliberately
does not restart the service.

## Operations and recovery

```bash
bin/backup-state.sh
```

This backs up the annotation database using SQLite's online backup API, mirrors
audit events, and backs up the MLflow SQLite database. Install and enable the
included daily timer if desired:

```bash
sudo cp systemd/llama-router-backup.* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-router-backup.timer
```

The in-progress pre-refactor Qwen3.8 run must be archived as legacy lineage
after it exits, before deleting any old launcher:

```bash
.venv/bin/python3 src/data/manage.py archive-legacy \
  --mlflow-run-id 02c888383fcf47628039d6332113275c \
  --model Qwen3.8-2B-VoiceInk-v1 \
  --train datasets/qwen35-2b-voiceink-v3/train.jsonl \
  --eval datasets/regression/voiceink-locked-440.jsonl \
  --output training/legacy-runs/qwen38-2b-voiceink-v1.json
```



## Canonical and legacy filesystem boundaries

The runtime now distinguishes durable state from experiments by construction:

```text
datasets/registry/       durable annotation database and audit history
datasets/releases/       sealed, immutable training releases
datasets/regression/     locked 440-sample regression corpus
datasets/legacy/         historical prepared datasets and snapshots
training/canonical/      manifest-driven runs only
training/legacy/         historical checkpoints, adapters, and smoke runs
models/canonical/        production and promotable VoiceInk artifacts
models/legacy/           superseded VoiceInk artifacts
results/canonical/       complete promotion-gate evaluations
results/legacy/          historical and exploratory evaluations
```

Inspect the reversible migration plan at any time:

```bash
PYTHONPATH=src .venv/bin/python3 src/data/legacy_layout.py
```

Apply it only with no trainer running:

```bash
PYTHONPATH=src .venv/bin/python3 src/data/legacy_layout.py --apply
```

The command fails closed while training is active, moves rather than deletes,
rewrites model paths in `config/models.ini`, and records every source, target,
classification, and byte count in `datasets/registry/legacy-migrations/`.
Rerunning it is safe. The current Qwen3.8 job is archived first, then classified
as canonical; all older datasets, runs, VoiceInk models, and evals go under the
explicit legacy namespaces.

Legacy model-specific launchers remain only as low-level engines or historical
references. New jobs must enter through `src/training/train.py`; it rejects
unsealed manifests. New outputs are rooted in `training/canonical/`,
`models/canonical/`, and `results/canonical/`.
