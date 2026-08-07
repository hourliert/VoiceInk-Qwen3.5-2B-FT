"""Shared subprocess adapters for Claude CLI and Codex CLI text tasks."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_CODEX_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "low"
PROVIDERS = ("claude", "codex")
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")

CODEX_TASK_PREAMBLE = """You are a bounded text-processing worker.
Do not use tools, inspect files, browse, or modify anything. Work only from the
text supplied in this request and return only the requested final output.

"""


def add_provider_args(parser, *, prefix: str = "", default_provider: str = "claude") -> None:
    """Add provider/model/reasoning CLI options to an ArgumentParser."""
    option_prefix = f"{prefix}-" if prefix else ""
    destination_prefix = f"{prefix}_" if prefix else ""
    parser.add_argument(
        f"--{option_prefix}provider",
        dest=f"{destination_prefix}provider",
        choices=PROVIDERS,
        default=default_provider,
        help="LLM CLI provider (Claude remains available; Codex uses ephemeral sessions)",
    )
    parser.add_argument(
        f"--{option_prefix}model",
        dest=f"{destination_prefix}model",
        default=None,
        help="Provider model (defaults to claude-sonnet-4-6 or gpt-5.6-luna)",
    )
    parser.add_argument(
        f"--{option_prefix}reasoning-effort",
        dest=f"{destination_prefix}reasoning_effort",
        choices=REASONING_EFFORTS,
        default=DEFAULT_REASONING_EFFORT,
        help="Codex reasoning effort (default: low; ignored by Claude)",
    )


def resolve_model(provider: str, model: str | None) -> str:
    if model:
        return model
    if provider == "codex":
        return DEFAULT_CODEX_MODEL
    return DEFAULT_CLAUDE_MODEL


def provider_metadata(provider: str, model: str, reasoning_effort: str) -> dict:
    metadata = {"provider": provider, "model": model}
    if provider == "codex":
        metadata["reasoning_effort"] = reasoning_effort
        metadata["session"] = "ephemeral"
    return metadata


def provider_prompt_path(claude_path: Path, provider: str) -> Path:
    """Resolve a provider-specific prompt while preserving legacy Claude files."""
    if provider == "claude":
        return claude_path
    return claude_path.with_name(f"{claude_path.stem}.codex{claude_path.suffix}")


def call_llm(
    prompt: str,
    *,
    provider: str,
    model: str,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    output_schema: dict | None = None,
    timeout: int = 600,
) -> str:
    """Call a supported CLI and return its final response text.

    Codex prompts are supplied over stdin and run in a temporary, read-only,
    non-persisted workspace. Claude retains the repository's previous CLI
    behavior for backwards compatibility.
    """
    if provider == "claude":
        return _call_claude(prompt, model=model, timeout=timeout)
    if provider == "codex":
        return _call_codex(
            prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            output_schema=output_schema,
            timeout=timeout,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _call_claude(prompt: str, *, model: str, timeout: int) -> str:
    env = {key: value for key, value in os.environ.items() if key != "CLAUDECODE"}
    env["CLAUDE_CODE_SKIP_UPDATE_CHECK"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes"
    result = subprocess.run(
        [
            "claude",
            "-p", prompt,
            "--model", model,
            "--disable-slash-commands",
            "--allowed-tools", "",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _call_codex(
    prompt: str,
    *,
    model: str,
    reasoning_effort: str,
    output_schema: dict | None,
    timeout: int,
) -> str:
    with tempfile.TemporaryDirectory(prefix="voiceink-codex-") as temporary:
        workdir = Path(temporary)
        output_path = workdir / "last-message.txt"
        command = [
            "codex", "exec",
            "--model", model,
            "--ephemeral",
            "--sandbox", "read-only",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
            "--color", "never",
            "--output-last-message", str(output_path),
            "--cd", str(workdir),
            "--config", f'model_reasoning_effort="{reasoning_effort}"',
        ]
        if output_schema is not None:
            schema_path = workdir / "output-schema.json"
            schema_path.write_text(
                json.dumps(output_schema, ensure_ascii=False), encoding="utf-8"
            )
            command.extend(["--output-schema", str(schema_path)])
        command.append("-")

        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        result = subprocess.run(
            command,
            input=CODEX_TASK_PREAMBLE + prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"codex CLI failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        if not output_path.is_file():
            raise RuntimeError("codex CLI completed without writing a final response")
        return output_path.read_text(encoding="utf-8").strip()


def parse_json_response(raw: str) -> dict:
    """Parse a JSON object, tolerating legacy providers' Markdown fences."""
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    parsed = json.loads(cleaned.strip())
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from LLM provider")
    return parsed


LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "minLength": 1}},
    "required": ["label"],
    "additionalProperties": False,
}

VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["pass", "fail"]},
        "type": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["status", "type", "reason"],
    "additionalProperties": False,
}

SCORE_DIMENSIONS = (
    "meaning_preservation",
    "filler_removal",
    "grammar_fluency",
    "technical_accuracy",
    "conciseness",
    "instruction_following",
)

SCORES_SCHEMA = {
    "type": "object",
    "properties": {
        dimension: {"type": "integer", "minimum": 1, "maximum": 5}
        for dimension in SCORE_DIMENSIONS
    },
    "required": list(SCORE_DIMENSIONS),
    "additionalProperties": False,
}

EVAL_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "output_a": SCORES_SCHEMA,
        "output_b": SCORES_SCHEMA,
    },
    "required": ["output_a", "output_b"],
    "additionalProperties": False,
}

SYNTHETIC_SCHEMA = {
    "type": "object",
    "properties": {
        "raw_transcript": {"type": "string", "minLength": 1},
        "clean_transcript": {"type": "string", "minLength": 1},
    },
    "required": ["raw_transcript", "clean_transcript"],
    "additionalProperties": False,
}
