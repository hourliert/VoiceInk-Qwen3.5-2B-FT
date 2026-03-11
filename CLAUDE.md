# llama-router

## Purpose

Local LLM inference server for improving raw speech-to-text transcriptions from VoiceInk (a macOS dictation app). VoiceInk sends raw transcripts to this server, which runs an open-source LLM to clean up and enhance the text, then returns the result.

## Architecture

- **Host machine**: Remote gaming PC (powerful GPU) running Linux, accessible from the Mac over the network.
- **LLM backend**: `llama.cpp` (`llama-server`) running locally on port 8002 (bound to 127.0.0.1).
- **Proxy**: A lightweight Python reverse proxy (`src/voiceink_proxy/server.py`) on port 8001 (bound to 0.0.0.0) that forwards VoiceInk requests to llama-server and logs every request/response pair as JSONL for future analysis.
- **Startup**: `bin/start.sh` launches both llama-server and the proxy. A systemd unit (`systemd/llama-router.service`) runs `start.sh` on boot.
- **Model config**: `config/models.ini` defines available models and shared inference parameters. Currently configured for several Qwen 3.5 variants (2B, 4B, 9B, 35B-A3B) at different quantization levels.

## Current model

The production model for the VoiceInk transcription cleanup task is **Qwen 3.5 4B** (Q4_K_XL quantization). It achieves ~150 tokens/second on this hardware. Thinking mode is disabled (`enable_thinking: false`) since the task is straightforward text cleanup.

## Fine-tuning goal

The 4B model works well but latency could be improved. The goal is to fine-tune a **Qwen 3.5 2B** model specifically for the transcription cleanup task using Unsloth, to get comparable quality at higher throughput. Previous attempts with the 2B model using prompt engineering alone were insufficient -- it didn't follow instructions well enough.

Training data is being collected via the proxy's JSONL logs (`logs/voiceink_proxy_requests.jsonl`).

## Key files

- `bin/start.sh` -- entrypoint, starts llama-server + proxy
- `config/models.ini` -- model definitions and inference params
- `src/voiceink_proxy/server.py` -- reverse proxy with JSONL logging
- `systemd/llama-router.service` -- systemd unit file
- `models/` -- GGUF model files (gitignored, ~30GB+ total)
- `logs/` -- request logs (gitignored)

## Development notes

- Python scripts use only the standard library (no pip dependencies).
- The llama.cpp binary lives at `/home/thomas/llama.cpp/llama-server` (outside this repo).
- VoiceInk on the Mac is already configured to send requests to this PC's IP on port 8001.
