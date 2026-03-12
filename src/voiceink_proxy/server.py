#!/usr/bin/env python3
import argparse
import http.client
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_body(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def try_parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_model(payload):
    return payload.get("model", "") if isinstance(payload, dict) else ""


def extract_response_text(payload):
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def do_PUT(self) -> None:
        self._proxy()

    def do_PATCH(self) -> None:
        self._proxy()

    def do_DELETE(self) -> None:
        self._proxy()

    def do_OPTIONS(self) -> None:
        self._proxy()

    def do_HEAD(self) -> None:
        self._proxy()

    def log_message(self, format: str, *args) -> None:
        return

    # Stop sequences injected into every chat completion request.
    # Prevents Qwen 3.5 thinking-mode leakage when thinking is disabled.
    INJECTED_STOP_TOKENS = ["</think>"]

    def _proxy(self) -> None:
        started = time.perf_counter()
        request_id = str(uuid.uuid4())

        request_body = b""
        content_length = self.headers.get("Content-Length")
        if content_length:
            request_body = self.rfile.read(int(content_length))

        # Inject stop tokens into chat completion requests
        if self.path == "/v1/chat/completions" and request_body:
            request_json = try_parse_json(decode_body(request_body))
            if isinstance(request_json, dict):
                existing = request_json.get("stop") or []
                if isinstance(existing, str):
                    existing = [existing]
                merged = list(dict.fromkeys(existing + self.INJECTED_STOP_TOKENS))
                request_json["stop"] = merged
                request_body = json.dumps(request_json, ensure_ascii=False).encode("utf-8")

        backend_headers = {}
        for key, value in self.headers.items():
            if key.lower() in HOP_BY_HOP_HEADERS:
                continue
            if key.lower() == "host":
                backend_headers[key] = f"{self.server.backend_host}:{self.server.backend_port}"
            else:
                backend_headers[key] = value
        # Update Content-Length after potential body modification
        if request_body:
            backend_headers["Content-Length"] = str(len(request_body))

        backend = http.client.HTTPConnection(
            self.server.backend_host,
            self.server.backend_port,
            timeout=self.server.backend_timeout,
        )

        response_body = b""
        response_status = 502
        response_reason = "Bad Gateway"
        response_headers = {}
        error_message = ""

        try:
            backend.request(
                self.command,
                self.path,
                body=request_body if request_body else None,
                headers=backend_headers,
            )
            response = backend.getresponse()
            response_status = response.status
            response_reason = response.reason
            response_body = response.read()
            response_headers = {key: value for key, value in response.getheaders()}
        except Exception as exc:
            error_message = str(exc)
            response_body = json.dumps({"error": error_message}).encode("utf-8")
            response_headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": str(len(response_body)),
            }
        finally:
            backend.close()

        self.send_response(response_status, response_reason)
        sent_content_length = False
        for key, value in response_headers.items():
            if key.lower() in HOP_BY_HOP_HEADERS:
                continue
            if key.lower() == "content-length":
                sent_content_length = True
            self.send_header(key, value)
        if not sent_content_length:
            self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(response_body)

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        raw_request = decode_body(request_body)
        raw_response = decode_body(response_body)
        request_json = try_parse_json(raw_request)
        response_json = try_parse_json(raw_response)

        self.server.logger.write(
            {
                "timestamp": utc_timestamp(),
                "request_id": request_id,
                "client_ip": self.client_address[0],
                "method": self.command,
                "path": self.path,
                "status_code": response_status,
                "duration_ms": duration_ms,
                "model": extract_model(request_json) or extract_model(response_json),
                "request_json_valid": request_json is not None,
                "response_json_valid": response_json is not None,
                "raw_request_json": raw_request,
                "raw_response_json": raw_response,
                "response_text": extract_response_text(response_json),
                "error": error_message,
            }
        )


class ProxyServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address,
        handler_class,
        *,
        backend_host: str,
        backend_port: int,
        backend_timeout: float,
        logger: JsonlLogger,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.backend_timeout = backend_timeout
        self.logger = logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reverse proxy for llama-server with structured JSONL logging.")
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8001)
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8002)
    parser.add_argument("--backend-timeout", type=float, default=300.0)
    parser.add_argument(
        "--log-file",
        default="/home/thomas/srv/llama-router/logs/voiceink_proxy_requests.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ProxyServer(
        (args.listen_host, args.listen_port),
        ProxyHandler,
        backend_host=args.backend_host,
        backend_port=args.backend_port,
        backend_timeout=args.backend_timeout,
        logger=JsonlLogger(Path(args.log_file)),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
