#!/usr/bin/env python3
"""Web UI for reviewing failed label validations.

Serves a single-page app that shows failed validation records from
datasets/labeled.jsonl, allowing the reviewer to approve or edit labels
without manually SSH-ing into the file.

Usage:
    python3 src/labeling/review_server.py
    python3 src/labeling/review_server.py --port 9000
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.extract import extract_from_record
from labeling.validate import LabeledDataset

DEFAULT_INPUT = ROOT / "datasets" / "labeled.jsonl"
DEFAULT_PORT = 8003


def get_failures(dataset):
    """Return enriched failure records (not manually reviewed)."""
    failures = []
    for record in dataset.records():
        v = record.get("validation", {})
        if v.get("status") == "fail" and not record.get("manually_reviewed"):
            try:
                components = extract_from_record(record)
            except (json.JSONDecodeError, KeyError):
                components = {
                    "transcript": "(extraction failed)",
                    "custom_vocabulary": "",
                    "clipboard_context": "",
                    "window_context": "",
                }
            failures.append({
                "request_id": record["request_id"],
                "timestamp": record.get("timestamp", ""),
                "label": record.get("label", ""),
                "original_response": record.get("original_response", ""),
                "validation": record.get("validation", {}),
                "transcript": components["transcript"],
                "vocabulary": components["custom_vocabulary"],
                "clipboard": components["clipboard_context"],
                "window": components["window_context"],
            })
    return failures


class ReviewHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self._send_html(HTML_PAGE)
        elif self.path == "/api/failures":
            failures = get_failures(self.server.dataset)
            self._send_json(failures)
        else:
            self._send_error(404, "Not found")

    def do_POST(self):
        if self.path.startswith("/api/approve/"):
            request_id = self.path[len("/api/approve/"):]
            dataset = self.server.dataset
            with dataset._lock:
                if request_id in dataset._records:
                    dataset._records[request_id]["manually_reviewed"] = True
                    dataset._flush()
                    self._send_json({"ok": True})
                else:
                    self._send_error(404, "Record not found")

        elif self.path.startswith("/api/update/"):
            request_id = self.path[len("/api/update/"):]
            body = self._read_body()
            if body is None:
                return
            new_label = body.get("label", "")
            if not new_label.strip():
                self._send_error(400, "Label cannot be empty")
                return
            dataset = self.server.dataset
            with dataset._lock:
                if request_id in dataset._records:
                    dataset._records[request_id]["label"] = new_label
                    dataset._records[request_id]["manually_reviewed"] = True
                    dataset._flush()
                    self._send_json({"ok": True})
                else:
                    self._send_error(404, "Record not found")
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
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
.empty { text-align: center; padding: 80px 20px; color: #888; font-size: 18px; }
.failure-banner { background: #2c1a1a; border: 1px solid #5c2a2a; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; }
.failure-type { color: #e74c3c; font-weight: 700; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; }
.failure-reason { color: #ccc; margin-top: 4px; font-size: 14px; line-height: 1.5; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.panel { background: #1a1a2e; border: 1px solid #2a2a3e; border-radius: 8px; overflow: hidden; }
.panel-header { background: #22223a; padding: 8px 14px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #888; border-bottom: 1px solid #2a2a3e; }
.panel-body { padding: 14px; font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; font-size: 13px; line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; min-height: 120px; color: #ddd; }
textarea.panel-body { width: 100%; border: none; background: transparent; color: #ddd; resize: vertical; outline: none; }
textarea.panel-body:focus { background: #1e1e32; }
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

  app.innerHTML = `
    <div class="id-text">${r.request_id} &mdash; ${r.timestamp || ''}</div>
    <div class="failure-banner">
      <div class="failure-type">${esc(v.type || 'UNKNOWN')}</div>
      <div class="failure-reason">${esc(v.reason || '')}</div>
    </div>
    <div class="panels">
      <div class="panel">
        <div class="panel-header">Raw Transcript</div>
        <div class="panel-body">${esc(r.transcript)}</div>
      </div>
      <div class="panel">
        <div class="panel-header">Label (editable)</div>
        <textarea class="panel-body" id="label-editor">${esc(r.label)}</textarea>
      </div>
    </div>
    <div class="original-response">
      <div class="panel-header">Original Model Response</div>
      <div class="panel-body">${esc(r.original_response)}</div>
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
      <button class="btn btn-skip" onclick="skip(1)">Skip</button>
      <div class="shortcuts">
        <kbd>a</kbd> approve &nbsp; <kbd>s</kbd> save &nbsp; <kbd>n</kbd> next &nbsp; <kbd>p</kbd> prev
      </div>
    </div>
  `;
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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Labeled JSONL file")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    dataset = LabeledDataset(args.input)
    failures = [r for r in dataset.records()
                if r.get("validation", {}).get("status") == "fail"
                and not r.get("manually_reviewed")]
    print(f"Loaded {len(dataset)} records ({len(failures)} unreviewed failures)")

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", args.port), ReviewHandler)
    server.dataset = dataset

    print(f"Review UI: http://0.0.0.0:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
