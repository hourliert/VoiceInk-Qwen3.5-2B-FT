# Repository Guidelines

## Project Structure & Module Organization

This repository is a script-driven Python pipeline for VoiceInk transcript cleanup and Qwen fine-tuning. Core code lives under `src/`: `voiceink_proxy/server.py` runs the OpenAI-compatible reverse proxy, `common/extract.py` contains shared parsing helpers, and `labeling/`, `synthetic/`, `training/`, and `eval/` hold the data, generation, fine-tuning, and evaluation workflows. Operational files live in `bin/start.sh`, `config/models.ini`, `systemd/llama-router.service`, `Modelfile`, and `docs/`.

Generated or private artifacts are intentionally ignored: `logs/`, `datasets/*.jsonl`, `models/`, `results/`, `training/lora/`, `training/outputs/`, and model binaries such as `*.gguf`.

## Build, Test, and Development Commands

- `bin/start.sh`: starts `llama-server` on port `8002` and the proxy on port `8001`; set `LLAMA_SERVER` if the binary is not at the default path.
- `python3 src/labeling/label.py --parallel 5`: labels logged requests with Claude as judge.
- `python3 src/labeling/validate.py --show-failures`: reviews failed or suspicious labels.
- `python3 src/synthetic/generate.py --count 160 --parallel 5`: creates synthetic long-form samples.
- `python3 src/training/prepare_dataset.py --extra-input datasets/synthetic/labeled.jsonl`: converts labels to training/eval splits.
- `.venv/bin/python3 src/training/finetune.py --epochs 2 --export-gguf q4_k_m`: runs LoRA fine-tuning and GGUF export.
- `python3 src/eval/evaluate.py --baseline Qwen3.5-4B --candidate Qwen3.5-2B-VoiceInk`: runs blind A/B evaluation.

## Coding Style & Naming Conventions

Use Python 3.12+ and prefer the standard library unless a workflow explicitly requires GPU/training dependencies. Follow existing style: 4-space indentation, `snake_case` functions and variables, `PascalCase` classes, and uppercase module constants. Keep scripts CLI-friendly with `argparse`, `pathlib.Path`, UTF-8 file handling, and newline-delimited JSON for records.

## Testing Guidelines

There is no dedicated pytest suite today. Validate changes by running the smallest relevant pipeline step, using flags such as `--limit`, `--force`, `--resume`, or `--show-failures` where available. For new reusable logic, add focused tests under `tests/` with `test_*.py` names and avoid relying on local model files or private JSONL logs.

## Commit & Pull Request Guidelines

Recent history uses short imperative commit subjects, for example `Add Ollama model usage guide...`, `Improve label validator...`, and `Filter training data...`. Keep commits focused on one workflow or behavior change. Pull requests should describe the pipeline impact, list commands run, mention data/model artifacts intentionally omitted, and link related issues or evaluation results when relevant.

## Security & Configuration Tips

Do not commit raw VoiceInk logs, labeled datasets, model checkpoints, API keys, or machine-specific paths. Keep deploy changes in `config/models.ini`, `bin/start.sh`, and `systemd/llama-router.service` explicit and documented.
