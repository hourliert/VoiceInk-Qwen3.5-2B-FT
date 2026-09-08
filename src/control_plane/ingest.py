"""Crash-safe incremental ingestion of the append-only VoiceInk proxy log."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from labeling.annotation_store import AnnotationStore, utc_now
except ModuleNotFoundError:  # Imported as src.control_plane during unit tests.
    from src.labeling.annotation_store import AnnotationStore, utc_now


@dataclass
class IngestionStatus:
    source_path: str
    running: bool = False
    initialized: bool = False
    source_inode: int = 0
    byte_offset: int = 0
    source_bytes: int = 0
    line_number: int = 0
    imported_records: int = 0
    malformed_records: int = 0
    lag_bytes: int = 0
    last_ingested_at: str = ""
    error: str = ""


class ProxyLogIngester:
    """Tail one JSONL log and persist progress after committed imports.

    Delivery is at least once. If the process stops after importing a row but
    before updating the cursor, the row is replayed; sample upserts make that
    replay safe. Incomplete final lines are never acknowledged.
    """

    def __init__(
        self,
        store: AnnotationStore,
        path: Path,
        *,
        poll_interval: float = 0.25,
        batch_lines: int = 500,
    ):
        self.store = store
        self.path = path.resolve()
        self.poll_interval = poll_interval
        self.batch_lines = batch_lines
        self.stop_event = threading.Event()
        self._status_lock = threading.Lock()
        self._status = IngestionStatus(source_path=str(self.path))
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="voiceink-proxy-ingester",
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def status(self) -> dict:
        with self._status_lock:
            return asdict(self._status)

    def _update_status(self, **values) -> None:
        with self._status_lock:
            for key, value in values.items():
                setattr(self._status, key, value)

    @staticmethod
    def _eligible(record: dict) -> bool:
        return bool(
            record.get("path") == "/v1/chat/completions"
            and record.get("request_json_valid")
            and record.get("response_json_valid")
            and record.get("status_code") == 200
        )

    def _cursor(self) -> dict | None:
        row = self.store.connection.execute(
            "SELECT * FROM ingestion_cursors WHERE source_path=?",
            (str(self.path),),
        ).fetchone()
        return dict(row) if row else None

    def _save_cursor(
        self,
        *,
        inode: int,
        offset: int,
        line_number: int,
        imported: int,
        malformed: int,
    ) -> None:
        self.store.connection.execute(
            "INSERT INTO ingestion_cursors("
            "source_path,source_inode,byte_offset,line_number,imported_records,"
            "malformed_records,updated_at) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(source_path) DO UPDATE SET "
            "source_inode=excluded.source_inode,byte_offset=excluded.byte_offset,"
            "line_number=excluded.line_number,"
            "imported_records=ingestion_cursors.imported_records+excluded.imported_records,"
            "malformed_records=ingestion_cursors.malformed_records+excluded.malformed_records,"
            "updated_at=excluded.updated_at",
            (
                str(self.path),
                inode,
                offset,
                line_number,
                imported,
                malformed,
                utc_now(),
            ),
        )

    def ingest_once(self) -> int:
        if not self.path.is_file():
            self._update_status(initialized=True, error="", source_bytes=0, lag_bytes=0)
            return 0

        stat = self.path.stat()
        cursor = self._cursor()
        same_file = bool(
            cursor
            and int(cursor["source_inode"]) == stat.st_ino
            and int(cursor["byte_offset"]) <= stat.st_size
        )
        offset = int(cursor["byte_offset"]) if same_file else 0
        line_number = int(cursor["line_number"]) if same_file else 0
        imported = malformed = processed = 0

        with self.path.open("rb") as stream:
            stream.seek(offset)
            while processed < self.batch_lines:
                line_start = stream.tell()
                raw_line = stream.readline()
                if not raw_line:
                    break
                if not raw_line.endswith(b"\n"):
                    stream.seek(line_start)
                    break
                offset = stream.tell()
                line_number += 1
                processed += 1
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    malformed += 1
                    continue
                if isinstance(record, dict) and self._eligible(record):
                    imported += int(self.store.import_proxy_record(record))

        self._save_cursor(
            inode=stat.st_ino,
            offset=offset,
            line_number=line_number,
            imported=imported,
            malformed=malformed,
        )
        latest_size = self.path.stat().st_size if self.path.is_file() else offset
        persisted = self._cursor() or {}
        self._update_status(
            initialized=True,
            source_inode=stat.st_ino,
            byte_offset=offset,
            source_bytes=latest_size,
            line_number=line_number,
            imported_records=int(persisted.get("imported_records", 0)),
            malformed_records=int(persisted.get("malformed_records", 0)),
            lag_bytes=max(0, latest_size - offset),
            last_ingested_at=utc_now() if processed else self._status.last_ingested_at,
            error="",
        )
        return imported

    def _run(self) -> None:
        self._update_status(running=True)
        while not self.stop_event.is_set():
            try:
                imported = self.ingest_once()
                if imported or self.status()["lag_bytes"]:
                    continue
            except Exception as exc:
                self._update_status(error=f"{type(exc).__name__}: {exc}")
            self.stop_event.wait(self.poll_interval)
        self._update_status(running=False)
