#!/usr/bin/env python3
"""Generate gold-standard labels for VoiceInk transcription samples using an LLM.

Reads raw request logs, sends the original VoiceInk input (system prompt + user
message) to a configured CLI provider, and writes the labeled dataset
to a JSONL file. Already-labeled request IDs are skipped.

Usage:
    python3 src/labeling/label.py --limit 10
    python3 src/labeling/label.py --limit 50 --parallel 4
    python3 src/labeling/label.py  # label everything
"""
import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common.extract import extract_from_record
from common.llm_cli import (
    LABEL_SCHEMA,
    add_provider_args,
    call_llm,
    parse_json_response,
    provider_metadata,
    provider_prompt_path,
    resolve_model,
)
DEFAULT_INPUT = ROOT / "logs" / "voiceink_proxy_requests.jsonl"
DEFAULT_OUTPUT = ROOT / "datasets" / "labeled.jsonl"
JUDGE_PROMPT_PATH = Path(__file__).resolve().parent / "judge_prompt.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label VoiceInk transcription samples using an LLM.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help=f"Input JSONL log file (default: {DEFAULT_INPUT})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output labeled JSONL file (default: {DEFAULT_OUTPUT})")
    p.add_argument("--limit", type=int, default=0,
                   help="Max samples to label (0 = all)")
    p.add_argument("--parallel", type=int, default=1,
                   help="Number of parallel CLI calls")
    add_provider_args(p)
    p.add_argument("--dry-run", action="store_true",
                   help="Print prompts without calling the configured provider")
    p.add_argument("--longest", type=int, default=0,
                   help="Pick the N longest transcripts (by response length)")
    p.add_argument("--ids", nargs="*", default=None,
                   help="Label only these specific request IDs (prefix match)")
    p.add_argument("--ids-file", type=Path, default=None,
                   help="Read request IDs from a text file or JSONL manifest")
    p.add_argument("--exclude-ids-file", type=Path, default=None,
                   help="Exclude request IDs listed in a text file or JSONL manifest")
    p.add_argument("--force", action="store_true",
                   help="Re-label even if already labeled (use with --ids)")
    p.add_argument("--shuffle", action="store_true",
                   help="Randomize the order of entries before labeling")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed used with --shuffle (default: 42)")
    p.add_argument("--reference-labels", type=Path, default=None,
                   help="Restrict calibration to IDs in this labeled JSONL and retain its label")
    return p.parse_args()


def load_logs(path: Path) -> list[dict]:
    """Load and parse the proxy request log."""
    records = []
    malformed = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if record.get("request_json_valid") and record.get("response_json_valid"):
                records.append(record)
    if malformed:
        print(
            f"WARNING: skipped {malformed} malformed JSONL "
            f"{'record' if malformed == 1 else 'records'} from {path}",
            file=sys.stderr,
        )
    return records


class LabeledDataset:
    """Thread-safe labeled dataset backed by a JSONL file.

    Enforces uniqueness by request_id: saving a record with an existing ID
    replaces the old entry. All reads and writes go through this class.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._records: dict[str, dict] = {}  # request_id -> record
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._records[record["request_id"]] = record
                except (json.JSONDecodeError, KeyError):
                    continue

    def _flush(self) -> None:
        """Rewrite the entire file from the in-memory dict."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for record in self._records.values():
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def labeled_ids(self) -> set[str]:
        with self._lock:
            return set(self._records.keys())

    def save(self, record: dict) -> None:
        """Save a labeled record, replacing any existing entry with the same request_id."""
        with self._lock:
            self._records[record["request_id"]] = record
            self._flush()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def build_prompt(record: dict, judge_template: str) -> str | None:
    """Build the judge prompt with structured context from a record.

    Extracts transcript, vocabulary, clipboard, and window context
    from the record and passes them as separate placeholders to the
    judge prompt template.
    """
    try:
        components = extract_from_record(record)
    except (json.JSONDecodeError, KeyError):
        return None

    if not components["transcript"]:
        return None

    return judge_template.format(
        transcript=components["transcript"],
        custom_vocabulary=components["custom_vocabulary"] or "(none provided)",
        clipboard_context=components["clipboard_context"] or "(empty)",
        window_context=components["window_context"] or "(empty)",
    )


def load_reference_labels(path: Path) -> dict[str, dict]:
    """Load calibration references keyed by request ID."""
    references = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("request_id") and record.get("label"):
                    references[record["request_id"]] = record
            except json.JSONDecodeError:
                continue
    return references


def load_id_prefixes(path: Path) -> list[str]:
    """Load request ID prefixes from newline text or a JSONL manifest."""
    prefixes = []
    seen = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                try:
                    prefix = str(json.loads(line)["request_id"]).strip()
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid ID manifest entry at {path}:{line_number}"
                    ) from exc
            else:
                prefix = line
            if prefix and prefix not in seen:
                prefixes.append(prefix)
                seen.add(prefix)
    return prefixes


def label_one(record: dict, provider: str, model: str, reasoning_effort: str,
              dry_run: bool, judge_template: str,
              reference: dict | None = None) -> dict | None:
    """Label a single record. Returns the labeled record or None on failure."""
    prompt = build_prompt(record, judge_template)
    if prompt is None:
        return None

    if dry_run:
        print(f"--- DRY RUN [{record['request_id']}] ---")
        print(prompt[:500])
        print("...")
        return None

    try:
        schema = LABEL_SCHEMA if provider == "codex" else None
        raw = call_llm(
            prompt,
            provider=provider,
            model=model,
            reasoning_effort=reasoning_effort,
            output_schema=schema,
            timeout=600,
        )
        label = parse_json_response(raw)["label"].strip() if schema else raw.strip()
        if not label:
            raise ValueError("provider returned an empty label")
    except Exception as exc:
        print(f"  ERROR [{record['request_id']}]: {exc}", file=sys.stderr)
        return None

    result = {
        "request_id": record["request_id"],
        "timestamp": record["timestamp"],
        "model_used_for_label": model,
        "label_provider": provider,
        "label_provider_metadata": provider_metadata(
            provider, model, reasoning_effort
        ),
        "original_model": record.get("model", ""),
        "raw_request_json": record.get("raw_request_json", ""),
        "original_response": record.get("response_text", ""),
        "label": label,
    }
    if reference is not None:
        result["reference_label"] = reference.get("label", "")
        result["reference_model_used_for_label"] = reference.get(
            "model_used_for_label", ""
        )
        result["reference_manually_reviewed"] = bool(
            reference.get("manually_reviewed")
        )
    return result


def main() -> None:
    args = parse_args()
    model = resolve_model(args.provider, args.model)

    if args.ids is not None and args.ids_file is not None:
        print("Use only one of --ids or --ids-file", file=sys.stderr)
        sys.exit(1)
    id_prefixes = args.ids
    if args.ids_file is not None:
        if not args.ids_file.is_file():
            print(f"ID file not found: {args.ids_file}", file=sys.stderr)
            sys.exit(1)
        try:
            id_prefixes = load_id_prefixes(args.ids_file)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(id_prefixes)} request IDs from {args.ids_file}")
    excluded_prefixes = []
    if args.exclude_ids_file is not None:
        if not args.exclude_ids_file.is_file():
            print(f"Exclusion ID file not found: {args.exclude_ids_file}", file=sys.stderr)
            sys.exit(1)
        try:
            excluded_prefixes = load_id_prefixes(args.exclude_ids_file)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(
            f"Loaded {len(excluded_prefixes)} excluded request IDs "
            f"from {args.exclude_ids_file}"
        )

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    logs = load_logs(args.input)
    print(f"Loaded {len(logs)} valid log entries from {args.input}")
    if excluded_prefixes:
        before = len(logs)
        logs = [
            record for record in logs
            if not any(
                record["request_id"].startswith(prefix)
                for prefix in excluded_prefixes
            )
        ]
        print(f"Excluded {before - len(logs)} entries from the calibration pool")

    dataset = LabeledDataset(args.output)
    labeled_ids = dataset.labeled_ids()
    if labeled_ids:
        print(f"Found {len(labeled_ids)} already-labeled entries in {args.output}")

    references = None
    if args.reference_labels:
        if not args.reference_labels.is_file():
            print(f"Reference labels not found: {args.reference_labels}", file=sys.stderr)
            sys.exit(1)
        references = load_reference_labels(args.reference_labels)
        logs = [record for record in logs if record.get("request_id") in references]
        print(f"Restricted calibration pool to {len(logs)} reference-labeled entries")

    # Filter by specific IDs (prefix match) -- skip dedup when --force is set
    if id_prefixes:
        pool = logs if args.force else [r for r in logs if r["request_id"] not in labeled_ids]
        to_label = [r for r in pool
                    if any(r["request_id"].startswith(prefix) for prefix in id_prefixes)]
        print(f"Filtered to {len(to_label)} entries matching --ids{' (force)' if args.force else ''}")
    else:
        to_label = [r for r in logs if r["request_id"] not in labeled_ids]
        print(f"{len(to_label)} entries need labeling")

    # Shuffle entries
    if args.shuffle:
        random.Random(args.seed).shuffle(to_label)
        print(f"Shuffled entries with seed {args.seed}")

    # Pick the N longest by response text length
    if args.longest > 0:
        to_label.sort(key=lambda r: len(r.get("response_text", "")), reverse=True)
        to_label = to_label[:args.longest]
        print(f"Selected {len(to_label)} longest entries")

    if args.limit > 0:
        to_label = to_label[:args.limit]
        print(f"Limiting to {len(to_label)} entries")

    if not to_label:
        print("Nothing to label.")
        return

    # Read template once, pass to all workers
    prompt_path = provider_prompt_path(JUDGE_PROMPT_PATH, args.provider)
    if not prompt_path.is_file():
        print(f"Provider prompt not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    judge_template = prompt_path.read_text(encoding="utf-8")
    print(f"Using {args.provider} prompt: {prompt_path}")

    labeled_count = 0
    error_count = 0

    if args.parallel <= 1:
        for i, record in enumerate(to_label, 1):
            print(f"[{i}/{len(to_label)}] Labeling {record['request_id']}...")
            reference = references.get(record["request_id"]) if references else None
            result = label_one(
                record, args.provider, model, args.reasoning_effort,
                args.dry_run, judge_template, reference,
            )
            if result:
                dataset.save(result)
                labeled_count += 1
            else:
                error_count += 1
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(
                    label_one, record, args.provider, model,
                    args.reasoning_effort, args.dry_run, judge_template,
                    references.get(record["request_id"]) if references else None,
                ): record
                for record in to_label
            }
            for i, future in enumerate(as_completed(futures), 1):
                record = futures[future]
                print(f"[{i}/{len(to_label)}] Done {record['request_id']}")
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"  ERROR: {exc}", file=sys.stderr)
                    error_count += 1
                    continue
                if result:
                    dataset.save(result)
                    labeled_count += 1
                else:
                    error_count += 1

    print(f"\nDone. Labeled: {labeled_count}, Errors: {error_count}, Total: {len(dataset)}")
    if labeled_count:
        print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
