"""Durable derived-artifact maintenance for control-plane mutations."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from labeling.annotation_store import AnnotationStore, utc_now
except ModuleNotFoundError:  # Imported as src.control_plane during unit tests.
    from src.labeling.annotation_store import AnnotationStore, utc_now


@dataclass
class MaintenanceStatus:
    running: bool = False
    pending: int = 0
    last_completed_at: str = ""
    last_backup: str = ""
    last_export: str = ""
    error: str = ""


class MaintenanceWorker:
    """Recoverable outbox for exports, audit mirrors, and online backups."""

    def __init__(
        self,
        store: AnnotationStore,
        *,
        approved_export: Path,
        audit_mirror: Path,
        backup_dir: Path,
        locked_eval: Path,
    ):
        self.store = store
        self.approved_export = approved_export
        self.audit_mirror = audit_mirror
        self.backup_dir = backup_dir
        self.locked_eval = locked_eval
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self._status_lock = threading.Lock()
        self._status = MaintenanceStatus()
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="control-plane-maintenance",
        )
        self._recover()

    def start(self) -> None:
        self.thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self.stop_event.set()
        self.wake_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def status(self) -> dict:
        with self._status_lock:
            return asdict(self._status)

    def _update_status(self, **values) -> None:
        with self._status_lock:
            for key, value in values.items():
                setattr(self._status, key, value)

    def _recover(self) -> None:
        now = utc_now()
        self.store.connection.execute(
            "UPDATE maintenance_jobs SET state='pending',error='recovered after restart',"
            "updated_at=? WHERE state='running' OR (state='failed' AND attempts<3)",
            (now,),
        )

    def schedule(self, decision_id: int) -> int:
        now = utc_now()
        cursor = self.store.connection.execute(
            "INSERT OR IGNORE INTO maintenance_jobs("
            "kind,dedupe_key,state,payload_json,created_at,updated_at"
            ") VALUES('refresh-derived',?,'pending',?,?,?)",
            (
                f"decision:{decision_id}",
                json.dumps({"decision_id": decision_id}, sort_keys=True),
                now,
                now,
            ),
        )
        self.wake_event.set()
        return int(cursor.lastrowid or 0)

    def _pending(self) -> list[dict]:
        return [
            dict(row)
            for row in self.store.connection.execute(
                "SELECT * FROM maintenance_jobs WHERE state='pending' ORDER BY id"
            )
        ]

    def _run(self) -> None:
        self._update_status(running=True)
        while not self.stop_event.is_set():
            jobs = self._pending()
            if not jobs:
                self._update_status(pending=0)
                self.wake_event.wait(1.0)
                self.wake_event.clear()
                continue
            ids = [int(job["id"]) for job in jobs]
            placeholders = ",".join("?" for _ in ids)
            now = utc_now()
            self.store.connection.execute(
                f"UPDATE maintenance_jobs SET state='running',attempts=attempts+1,"
                f"updated_at=? WHERE id IN ({placeholders})",
                (now, *ids),
            )
            self._update_status(pending=len(ids), error="")
            try:
                self.store.mirror_audit_events(self.audit_mirror)
                export = self.store.export_approved(
                    self.approved_export,
                    self.locked_eval,
                )
                backup = self.store.backup(self.backup_dir)
                completed_at = utc_now()
                self.store.connection.execute(
                    f"UPDATE maintenance_jobs SET state='completed',error='',updated_at=? "
                    f"WHERE id IN ({placeholders})",
                    (completed_at, *ids),
                )
                self._update_status(
                    pending=0,
                    last_completed_at=completed_at,
                    last_backup=str(backup),
                    last_export=str(export.get("output", "")),
                    error="",
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.store.connection.execute(
                    f"UPDATE maintenance_jobs SET "
                    f"state=CASE WHEN attempts<3 THEN 'pending' ELSE 'failed' END,"
                    f"error=?,updated_at=? WHERE id IN ({placeholders})",
                    (error, utc_now(), *ids),
                )
                pending = self.store.connection.execute(
                    "SELECT COUNT(*) FROM maintenance_jobs WHERE state='pending'"
                ).fetchone()[0]
                self._update_status(pending=int(pending), error=error)
                self.stop_event.wait(1.0)
        self._update_status(running=False)
