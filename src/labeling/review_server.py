#!/usr/bin/env python3
"""Web UI for human review of labeled transcript-cleanup records.

The queue can contain validator failures, conservatively suspicious passes,
every unreviewed record, or exact IDs from a manifest. The reviewer can
approve, edit, or reject records without manually modifying JSONL files.

Usage:
    python3 src/labeling/review_server.py
    python3 src/labeling/review_server.py --mode suspicious --host 0.0.0.0
    python3 src/labeling/review_server.py --input pilot.jsonl --input batch.jsonl
    python3 src/labeling/review_server.py --ids-file datasets/review.jsonl
"""
import argparse
import difflib
import html
import json
import re
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.extract import extract_from_record
from labeling.validate import LabeledDataset

DEFAULT_INPUT = ROOT / "datasets" / "labeled.jsonl"
DEFAULT_PORT = 8003
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*%?\b")
OPENER_RE = re.compile(r"^(okay|ok|yeah|so|alright)[,!. ]", re.IGNORECASE)
WORD_RE = re.compile(r"[\w']+", re.UNICODE)
DIFF_TOKEN_RE = re.compile(r"\s+|[\w']+|[^\w\s]", re.UNICODE)


class ReviewDatasets:
    """Expose several labeled JSONL files as one safely writable collection."""

    def __init__(self, paths: list[Path]):
        self.datasets = [LabeledDataset(path) for path in paths]
        self._owners = {}
        for dataset in self.datasets:
            for record in dataset.records():
                request_id = record["request_id"]
                if request_id in self._owners:
                    raise ValueError(
                        f"Duplicate request ID across review inputs: {request_id}"
                    )
                self._owners[request_id] = dataset

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def records(self):
        for dataset in self.datasets:
            yield from dataset.records()

    def dataset_for(self, request_id: str):
        return self._owners.get(request_id)


def load_review_manifest(path: Path) -> dict[str, str]:
    """Load exact request IDs and optional reasons from text or JSONL."""
    selected = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                try:
                    entry = json.loads(line)
                    request_id = str(entry["request_id"]).strip()
                    reason = str(entry.get("reason", "selected for review")).strip()
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid review manifest entry at {path}:{line_number}"
                    ) from exc
            else:
                request_id = line
                reason = "selected for review"
            if request_id:
                selected[request_id] = reason
    return selected


def word_similarity(source: str, target: str) -> float:
    source_words = WORD_RE.findall(source.casefold())
    target_words = WORD_RE.findall(target.casefold())
    return difflib.SequenceMatcher(
        None, source_words, target_words, autojunk=False
    ).ratio()


def highlighted_diff(source: str, target: str) -> str:
    """Render an escaped inline lexical diff from source to target."""
    source_tokens = DIFF_TOKEN_RE.findall(source)
    target_tokens = DIFF_TOKEN_RE.findall(target)
    # Visual-only diff: autojunk avoids quadratic behavior on long, repetitive
    # transcripts. Approval similarity uses word_similarity() instead.
    matcher = difflib.SequenceMatcher(None, source_tokens, target_tokens)
    rendered = []
    for operation, source_start, source_end, target_start, target_end in (
        matcher.get_opcodes()
    ):
        source_text = html.escape("".join(source_tokens[source_start:source_end]))
        target_text = html.escape("".join(target_tokens[target_start:target_end]))
        if operation == "equal":
            rendered.append(target_text)
        elif operation == "delete":
            rendered.append(f'<del class="diff-remove">{source_text}</del>')
        elif operation == "insert":
            rendered.append(f'<ins class="diff-add">{target_text}</ins>')
        else:
            rendered.append(f'<del class="diff-remove">{source_text}</del>')
            rendered.append(f'<ins class="diff-add">{target_text}</ins>')
    return "".join(rendered)


def suspicion_reasons(record: dict, components: dict) -> list[str]:
    """Return conservative deterministic reasons to human-review a validator pass."""
    raw = components["transcript"]
    label = str(record.get("label", ""))
    reasons = []
    if not raw or not label:
        return ["missing transcript or label"]

    ratio = len(label) / max(1, len(raw))
    similarity = word_similarity(raw, label)
    if ratio < 0.78:
        reasons.append(f"short label/raw ratio {ratio:.2f}")
    elif ratio > 1.12:
        reasons.append(f"long label/raw ratio {ratio:.2f}")
    if similarity < 0.35:
        reasons.append(f"low raw/label similarity {similarity:.2f}")
    if set(NUMBER_RE.findall(raw)) != set(NUMBER_RE.findall(label)):
        reasons.append("numeric tokens changed")
    if len(raw) >= 4000:
        reasons.append("long transcript requires human spot-check")
    if OPENER_RE.search(label):
        reasons.append("possible retained discourse opener")
    if "<unk>" in label.casefold():
        reasons.append("unresolved unknown token")
    return reasons


def review_is_complete(record: dict) -> bool:
    auto_approved = (record.get("auto_review") or {}).get("status") == "approved"
    return auto_approved or bool(record.get("manually_reviewed")) or (
        (record.get("manual_review") or {}).get("status") == "rejected"
    )


def get_review_records(dataset, mode="failures", selected=None):
    """Return enriched, incomplete records for the configured review queue."""
    review_records = []
    selected = selected or {}
    for record in dataset.records():
        request_id = record["request_id"]
        if review_is_complete(record):
            continue
        v = record.get("validation", {})
        try:
            components = extract_from_record(record)
        except (json.JSONDecodeError, KeyError):
            components = {
                "transcript": "(extraction failed)",
                "custom_vocabulary": "",
                "clipboard_context": "",
                "window_context": "",
            }

        if selected:
            if request_id not in selected:
                continue
            queue_reasons = [selected[request_id]]
        elif mode == "all-unreviewed":
            queue_reasons = ["all unreviewed records requested"]
        elif v.get("status") == "fail":
            queue_reasons = ["validator failure"]
        elif mode == "suspicious":
            if v.get("status") != "pass":
                queue_reasons = ["missing validation result"]
            else:
                queue_reasons = suspicion_reasons(record, components)
                if not queue_reasons:
                    continue
        else:
            continue

        transcript = components["transcript"]
        original_response = str(
            record.get("current_model_response")
            or record.get("original_response", "")
        )
        original_model = str(
            record.get("current_model")
            or record.get("original_model")
            or "Original production model"
        )
        reference_label = record.get("reference_label")
        if reference_label is not None:
            reference_label = str(reference_label)
        label = str(record.get("label", ""))
        enriched = {
            "request_id": request_id,
            "timestamp": record.get("timestamp", ""),
            "label": label,
            "original_response": original_response,
            "original_model": original_model,
            "reference_label": reference_label,
            "reference_model": record.get(
                "reference_model_used_for_label", "reference model"
            ),
            "validation": v,
            "queue_reason": "; ".join(queue_reasons),
            "transcript": transcript,
            "original_diff_html": highlighted_diff(
                transcript, original_response
            ),
            "label_diff_html": highlighted_diff(transcript, label),
            "original_similarity": round(
                word_similarity(transcript, original_response) * 100
            ),
            "label_similarity": round(
                word_similarity(transcript, label) * 100
            ),
            "label_original_similarity": round(
                word_similarity(original_response, label) * 100
            ),
            "vocabulary": components["custom_vocabulary"],
            "clipboard": components["clipboard_context"],
            "window": components["window_context"],
        }
        if reference_label is not None:
            enriched.update({
                "reference_diff_html": highlighted_diff(
                    transcript, reference_label
                ),
                "reference_similarity": round(
                    word_similarity(transcript, reference_label) * 100
                ),
                "label_reference_similarity": round(
                    word_similarity(reference_label, label) * 100
                ),
            })
        review_records.append(enriched)
    return review_records


class ReviewHandler(BaseHTTPRequestHandler):
    def _dataset_for(self, request_id):
        collection = self.server.dataset
        if hasattr(collection, "dataset_for"):
            return collection.dataset_for(request_id)
        if request_id in collection._records:
            return collection
        return None

    def do_GET(self):
        if self.path == "/":
            self._send_html(HTML_PAGE)
        elif self.path == "/api/failures":
            records = get_review_records(
                self.server.dataset,
                mode=self.server.review_mode,
                selected=self.server.selected_records,
            )
            self._send_json(records)
        else:
            self._send_error(404, "Not found")

    def do_POST(self):
        if self.path.startswith("/api/approve/"):
            request_id = self.path[len("/api/approve/"):]
            dataset = self._dataset_for(request_id)
            if dataset is None:
                self._send_error(404, "Record not found")
                return
            with dataset._lock:
                record = dataset._records[request_id]
                record.pop("auto_review", None)
                record["manually_reviewed"] = True
                record["manual_review"] = {
                    "status": "approved",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                dataset._flush()
                self._send_json({"ok": True})

        elif self.path.startswith("/api/update/"):
            request_id = self.path[len("/api/update/"):]
            body = self._read_body()
            if body is None:
                return
            new_label = body.get("label", "")
            if not new_label.strip():
                self._send_error(400, "Label cannot be empty")
                return
            dataset = self._dataset_for(request_id)
            if dataset is None:
                self._send_error(404, "Record not found")
                return
            with dataset._lock:
                record = dataset._records[request_id]
                record.pop("auto_review", None)
                record["label"] = new_label.strip()
                record["manually_reviewed"] = True
                record["manual_review"] = {
                    "status": "edited",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                dataset._flush()
                self._send_json({"ok": True})
        elif self.path.startswith("/api/reject/"):
            request_id = self.path[len("/api/reject/"):]
            dataset = self._dataset_for(request_id)
            if dataset is None:
                self._send_error(404, "Record not found")
                return
            with dataset._lock:
                record = dataset._records[request_id]
                record.pop("auto_review", None)
                record.pop("manually_reviewed", None)
                record["manual_review"] = {
                    "status": "rejected",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                dataset._flush()
                self._send_json({"ok": True})
        else:
            self._send_error(404, "Not found")

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            return json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            self._send_error(400, "Invalid JSON")
            return None

    def _send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code, message):
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quieter logging — only show POST actions
        if "POST" in (args[0] if args else ""):
            super().log_message(format, *args)


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label Review</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #0f0f17; color: #e0e0e0; min-height: 100vh; }
.header { background: #1a1a2e; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #2a2a3e; }
.header h1 { font-size: 18px; font-weight: 600; color: #fff; }
.badge { background: #e74c3c; color: #fff; padding: 3px 10px; border-radius: 12px; font-size: 13px; font-weight: 600; }
.badge.done { background: #27ae60; }
.container { max-width: 1800px; margin: 0 auto; padding: 20px; }
.empty { text-align: center; padding: 80px 20px; color: #888; font-size: 18px; }
.failure-banner { background: #2c1a1a; border: 1px solid #5c2a2a; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; }
.failure-type { color: #e74c3c; font-weight: 700; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; }
.failure-reason { color: #ccc; margin-top: 4px; font-size: 14px; line-height: 1.5; }
.panels { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; align-items: start; }
.panels.four-way { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.panel { background: #1a1a2e; border: 1px solid #2a2a3e; border-radius: 8px; overflow: hidden; }
.panel-header { background: #22223a; padding: 8px 14px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #aaa; border-bottom: 1px solid #2a2a3e; display: flex; justify-content: space-between; gap: 8px; }
.panel-body { padding: 14px; font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; font-size: 13px; line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; min-height: 120px; color: #ddd; }
.match-score { color: #7f8caa; font-size: 10px; white-space: nowrap; }
.diff-legend { color: #777; font-size: 12px; margin: -6px 0 12px; }
.diff-remove { background: rgba(231, 76, 60, 0.22); color: #ff9b91; text-decoration: line-through; text-decoration-thickness: 1px; }
.diff-add { background: rgba(46, 204, 113, 0.22); color: #a5efbf; text-decoration: none; }
.editor-label { padding: 7px 14px; background: #19192a; border-top: 1px solid #2a2a3e; border-bottom: 1px solid #2a2a3e; color: #777; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
textarea.panel-body { width: 100%; height: auto; overflow-y: hidden; border: none; background: transparent; color: #ddd; resize: none; outline: none; }
textarea.panel-body:focus { background: #1e1e32; }
@media (max-width: 1050px) { .panels { grid-template-columns: 1fr; } }
.context-toggle { background: none; border: 1px solid #2a2a3e; border-radius: 6px; color: #888; padding: 6px 14px; cursor: pointer; font-size: 12px; margin-bottom: 12px; }
.context-toggle:hover { color: #bbb; border-color: #444; }
.context-section { display: none; margin-bottom: 16px; }
.context-section.open { display: block; }
.context-block { background: #1a1a2e; border: 1px solid #2a2a3e; border-radius: 8px; padding: 12px 14px; margin-bottom: 8px; }
.context-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #666; margin-bottom: 4px; }
.context-value { font-family: monospace; font-size: 12px; line-height: 1.5; white-space: pre-wrap; color: #aaa; }
.original-response { background: #1a1a2e; border: 1px solid #2a2a3e; border-radius: 8px; overflow: hidden; margin-bottom: 16px; }
.actions { display: flex; gap: 10px; align-items: center; }
.btn { padding: 10px 22px; border: none; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; transition: opacity 0.15s; }
.btn:hover { opacity: 0.85; }
.btn-approve { background: #27ae60; color: #fff; }
.btn-save { background: #2980b9; color: #fff; }
.btn-reject { background: #9b2c2c; color: #fff; }
.btn-skip { background: #333; color: #aaa; }
.shortcuts { margin-left: auto; color: #555; font-size: 12px; }
kbd { background: #222; border: 1px solid #444; border-radius: 3px; padding: 1px 5px; font-family: monospace; font-size: 11px; color: #888; }
.nav-info { color: #666; font-size: 13px; margin-bottom: 16px; }
.id-text { font-family: monospace; font-size: 12px; color: #666; margin-bottom: 16px; }
</style>
</head>
<body>
<div class="header">
  <h1>Label Review</h1>
  <span id="badge" class="badge">loading...</span>
</div>
<div class="container" id="app">
  <div class="empty">Loading...</div>
</div>

<script>
let records = [];
let idx = 0;

async function load() {
  const res = await fetch('/api/failures');
  records = await res.json();
  idx = 0;
  render();
}

function render() {
  const app = document.getElementById('app');
  const badge = document.getElementById('badge');

  if (records.length === 0) {
    badge.textContent = 'all clear';
    badge.className = 'badge done';
    app.innerHTML = '<div class="empty">No failures to review. All done!</div>';
    return;
  }

  badge.textContent = `${idx + 1} / ${records.length}`;
  badge.className = 'badge';

  const r = records[idx];
  const v = r.validation || {};
  const referencePanel = r.reference_label == null ? '' : `
      <div class="panel">
        <div class="panel-header">
          <span>Sonnet reference</span>
          <span class="match-score">${r.reference_similarity}% word match</span>
        </div>
        <div class="panel-body">${r.reference_diff_html}</div>
      </div>`;
  const referenceMatch = r.label_reference_similarity == null
    ? ''
    : ` · ${r.label_reference_similarity}% Sonnet`;

  app.innerHTML = `
    <div class="id-text">${r.request_id} &mdash; ${r.timestamp || ''}</div>
    <div class="failure-banner">
      <div class="failure-type">Review queue</div>
      <div class="failure-reason">${esc(r.queue_reason || '')}</div>
      <div class="failure-type">${esc(v.type || 'UNKNOWN')}</div>
      <div class="failure-reason">${esc(v.reason || 'Validator passed; selected by conservative checks.')}</div>
    </div>
    <div class="diff-legend">
      Comparison against raw: <span class="diff-remove">removed/replaced raw text</span>
      &nbsp; <span class="diff-add">added/replacement output text</span>
    </div>
    <div class="panels ${referencePanel ? 'four-way' : ''}">
      <div class="panel">
        <div class="panel-header">Raw Transcript</div>
        <div class="panel-body">${esc(r.transcript)}</div>
      </div>
      ${referencePanel}
      <div class="panel">
        <div class="panel-header">
          <span>${esc(r.original_model)}</span>
          <span class="match-score">${r.original_similarity}% word match</span>
        </div>
        <div class="panel-body">${r.original_diff_html}</div>
      </div>
      <div class="panel">
        <div class="panel-header">
          <span>New Luna label</span>
          <span class="match-score">${r.label_similarity}% raw · ${r.label_original_similarity}% Qwen${referenceMatch}</span>
        </div>
        <div class="panel-body">${r.label_diff_html}</div>
        <div class="editor-label">Editable label — full text</div>
        <textarea class="panel-body" id="label-editor">${esc(r.label)}</textarea>
      </div>
    </div>
    <button class="context-toggle" onclick="toggleContext()">Show context</button>
    <div class="context-section" id="context-section">
      <div class="context-block">
        <div class="context-label">Window Context</div>
        <div class="context-value">${esc(r.window || '(empty)')}</div>
      </div>
      <div class="context-block">
        <div class="context-label">Clipboard</div>
        <div class="context-value">${esc(r.clipboard || '(empty)')}</div>
      </div>
      <div class="context-block">
        <div class="context-label">Vocabulary</div>
        <div class="context-value">${esc(r.vocabulary || '(empty)')}</div>
      </div>
    </div>
    <div class="actions">
      <button class="btn btn-approve" onclick="approve()">Approve</button>
      <button class="btn btn-save" onclick="saveEdit()">Save Edit</button>
      <button class="btn btn-reject" onclick="rejectRecord()">Reject</button>
      <button class="btn btn-skip" onclick="skip(1)">Skip</button>
      <div class="shortcuts">
        <kbd>a</kbd> approve &nbsp; <kbd>s</kbd> save &nbsp; <kbd>r</kbd> reject &nbsp; <kbd>n</kbd> next &nbsp; <kbd>p</kbd> prev
      </div>
    </div>
  `;
  const editor = document.getElementById('label-editor');
  editor.addEventListener('input', autoSizeEditor);
  autoSizeEditor.call(editor);
}

function autoSizeEditor() {
  this.style.height = 'auto';
  this.style.height = `${this.scrollHeight}px`;
}

function esc(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function toggleContext() {
  const el = document.getElementById('context-section');
  el.classList.toggle('open');
  const btn = el.previousElementSibling;
  btn.textContent = el.classList.contains('open') ? 'Hide context' : 'Show context';
}

async function approve() {
  const r = records[idx];
  await fetch(`/api/approve/${r.request_id}`, { method: 'POST' });
  records.splice(idx, 1);
  if (idx >= records.length) idx = Math.max(0, records.length - 1);
  render();
}

async function saveEdit() {
  const r = records[idx];
  const editor = document.getElementById('label-editor');
  const newLabel = editor.value;
  await fetch(`/api/update/${r.request_id}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label: newLabel }),
  });
  records.splice(idx, 1);
  if (idx >= records.length) idx = Math.max(0, records.length - 1);
  render();
}

async function rejectRecord() {
  const r = records[idx];
  await fetch(`/api/reject/${r.request_id}`, { method: 'POST' });
  records.splice(idx, 1);
  if (idx >= records.length) idx = Math.max(0, records.length - 1);
  render();
}

function skip(dir) {
  if (records.length === 0) return;
  idx = (idx + dir + records.length) % records.length;
  render();
}

document.addEventListener('keydown', (e) => {
  // Don't intercept when typing in textarea
  if (e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'a') approve();
  else if (e.key === 's') saveEdit();
  else if (e.key === 'r') rejectRecord();
  else if (e.key === 'n' || e.key === 'ArrowRight') skip(1);
  else if (e.key === 'p' || e.key === 'ArrowLeft') skip(-1);
});

load();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Web UI for reviewing label validations")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (use 0.0.0.0 for LAN access)")
    parser.add_argument(
        "--input", dest="inputs", type=Path, action="append",
        help="Labeled JSONL file; repeat to review several files together",
    )
    parser.add_argument(
        "--mode", choices=("failures", "suspicious", "all-unreviewed"),
        default="failures", help="Which unreviewed records to queue",
    )
    parser.add_argument(
        "--ids-file", type=Path, default=None,
        help="Review these exact IDs regardless of validation status",
    )
    args = parser.parse_args()

    input_paths = args.inputs or [DEFAULT_INPUT]
    for path in input_paths:
        if not path.exists():
            print(f"Input file not found: {path}", file=sys.stderr)
            sys.exit(1)

    try:
        dataset = ReviewDatasets(input_paths)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    selected_records = {}
    if args.ids_file is not None:
        if not args.ids_file.is_file():
            print(f"ID file not found: {args.ids_file}", file=sys.stderr)
            sys.exit(1)
        try:
            selected_records = load_review_manifest(args.ids_file)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        available_ids = {record["request_id"] for record in dataset.records()}
        missing_ids = sorted(selected_records.keys() - available_ids)
        if missing_ids:
            print(f"Review IDs not found in input: {missing_ids}", file=sys.stderr)
            sys.exit(1)

    queued = get_review_records(
        dataset, mode=args.mode, selected=selected_records
    )
    print(f"Loaded {len(dataset)} records ({len(queued)} queued for review)")

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    server.dataset = dataset
    server.review_mode = args.mode
    server.selected_records = selected_records

    print(f"Review UI listening on http://{args.host}:{args.port}")
    if args.host == "0.0.0.0":
        print(f"Open http://<this-machine-LAN-IP>:{args.port} from another device")
        print("WARNING: the review UI has no authentication; use only on a trusted LAN")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
