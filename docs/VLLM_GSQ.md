# Qwen3.8 GSQ coding experiment

The Qwen3.8 27B GSQ checkpoint runs as a coding-only vLLM service. It does not
replace llama.cpp or participate in the VoiceInk proxy path.

| Service | Address | Purpose |
| --- | --- | --- |
| VoiceInk proxy | `0.0.0.0:8001` | VoiceInk logging and request handling |
| llama.cpp | `0.0.0.0:8002` | Existing on-demand GGUF models |
| Control plane | `192.168.1.150:8003` | Review and labeling workflows |
| vLLM GSQ | `127.0.0.1:8005` | Experimental Qwen3.8 coding backend |

## Install

Run the idempotent setup script:

```bash
bin/setup-vllm-gsq.sh
```

It downloads the checkpoint to `models/Qwen3.8-27B-3Bit-GSQ`, creates the
isolated `.venv-vllm`, installs vLLM 0.27.1, and applies the checkpoint's
required quantized-embedding patch.

## Start and inspect

`bin/start.sh` starts the vLLM backend by default as a non-fatal sibling
process. A vLLM failure therefore does not restart or stop VoiceInk. Disable
the experiment when needed with `VLLM_AUTOSTART=0`.

The standalone launcher is also available for diagnosis:

```bash
bin/start-vllm-gsq.sh
```

Useful checks:

```bash
curl http://127.0.0.1:8005/health
curl http://127.0.0.1:8005/v1/models
tail -f logs/vllm-gsq.log
```

Launcher defaults can be overridden with `VLLM_MAX_MODEL_LEN`,
`VLLM_MAX_NUM_SEQS`, and `VLLM_GPU_MEMORY_UTILIZATION`. The tested defaults
are a 65,536-token model limit, four scheduled sequences, FP8 KV cache, and 90%
GPU-memory utilization.

On the 16 GB RTX 4080 Super, a clean managed boot profiled 76,458 aggregate
KV-cache tokens.
This supports one full 65,536-token request or several smaller concurrent
requests; it does not support four independent 65,536-token contexts.

## Connect from a Mac

The experimental API is localhost-only because it has no authentication. Open
an SSH tunnel from the Mac:

```bash
ssh -N -L 8005:127.0.0.1:8005 thomas@192.168.1.150
```

Configure the coding agent with:

```text
base URL:      http://127.0.0.1:8005/v1
model:         Qwen3.8-27B-3Bit-GSQ
context limit: 65536
```

The server supports OpenAI chat completions, separated Qwen reasoning, and
structured tool calls through the `qwen3_coder` parser.
