import inspect
import json
import queue
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

from src.control_plane.ingest import ProxyLogIngester
from src.control_plane.maintenance import MaintenanceWorker
from src.control_plane.server import ControlPlaneServer
from src.data.cohort_review import cohort_status
from src.control_plane.state import ControlPlaneState
from src.data.schema import migrate
from src.labeling.annotation_store import AnnotationStore

from tests.test_live_review import proxy_record


class IncrementalIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = AnnotationStore(self.root / "annotations.sqlite3")
        self.addCleanup(self.store.close)
        migrate(self.store.connection)
        self.log = self.root / "proxy.jsonl"

    def test_partial_line_restart_and_rotation_are_safe(self) -> None:
        first = proxy_record()
        encoded = json.dumps(first).encode("utf-8")
        self.log.write_bytes(encoded)

        ingester = ProxyLogIngester(self.store, self.log)
        self.assertEqual(ingester.ingest_once(), 0)
        self.assertIsNone(self.store.sample(first["request_id"]))

        with self.log.open("ab") as stream:
            stream.write(b"\n")
        self.assertEqual(ingester.ingest_once(), 1)
        self.assertIsNotNone(self.store.sample(first["request_id"]))
        cursor = ingester.status()
        self.assertEqual(cursor["lag_bytes"], 0)

        restarted = ProxyLogIngester(self.store, self.log)
        self.assertEqual(restarted.ingest_once(), 0)

        rotated = self.root / "proxy.jsonl.1"
        self.log.rename(rotated)
        second = dict(first, request_id="request-after-rotation")
        self.log.write_text(json.dumps(second) + "\n", encoding="utf-8")
        self.assertEqual(restarted.ingest_once(), 1)
        self.assertIsNotNone(self.store.sample("request-after-rotation"))


class StoreConcurrencyTests(unittest.TestCase):
    def test_concurrent_writers_preserve_every_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AnnotationStore(Path(directory) / "annotations.sqlite3")
            migrate(store.connection)
            base = proxy_record()
            request_ids = []
            for index in range(20):
                request_id = "concurrent-" + str(index)
                request_ids.append(request_id)
                store.import_proxy_record(dict(base, request_id=request_id))

            def decide(request_id: str) -> None:
                annotation_id = store.create_annotation(
                    request_id,
                    "Corrected " + request_id,
                    origin="human",
                )
                store.decide(
                    request_id,
                    "human_edit",
                    annotation_id=annotation_id,
                )

            with ThreadPoolExecutor(max_workers=10) as executor:
                list(executor.map(decide, request_ids))

            current = store.connection.execute(
                "SELECT COUNT(*) FROM decisions WHERE is_current=1 "
                "AND request_id LIKE 'concurrent-%'"
            ).fetchone()[0]
            self.assertEqual(current, 20)
            self.assertEqual(store.integrity_check(), "ok")
            store.close()


class PureReadProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = AnnotationStore(self.root / "annotations.sqlite3")
        self.addCleanup(self.store.close)
        migrate(self.store.connection)
        self.store.import_proxy_record(proxy_record())
        annotation = self.store.create_annotation(
            "request-1", "Use Monza.", origin="historical"
        )
        cursor = self.store.connection.execute(
            "INSERT INTO canonical_cohorts(name,state,seed,specification_json,"
            "created_at,updated_at) VALUES(?,?,?,?,?,?)",
            ("cohort-v1", "reviewing", 1, "{}", "2026-01-01", "2026-01-01"),
        )
        self.store.connection.execute(
            "INSERT INTO canonical_cohort_members(cohort_id,request_id,annotation_id,"
            "split,stratum,audit_selected) VALUES(?,?,?,?,?,?)",
            (cursor.lastrowid, "request-1", annotation, "train", "representative", 0),
        )

    def test_cohort_status_is_set_based_and_read_only(self) -> None:
        statements = []
        self.store.connection.set_trace_callback(statements.append)
        before = self.store.connection.total_changes
        result = cohort_status(self.store.connection, "cohort-v1")
        after = self.store.connection.total_changes
        self.store.connection.set_trace_callback(None)

        self.assertEqual(result["members"], 1)
        self.assertEqual(after, before)
        self.assertLessEqual(len(statements), 3)
        self.assertFalse(any(statement.lstrip().upper().startswith(
            ("INSERT", "UPDATE", "DELETE")
        ) for statement in statements))

    def test_overview_does_not_call_external_or_ingestion_work(self) -> None:
        overview_source = inspect.getsource(ControlPlaneState.overview)
        self.assertNotIn("self.runs(", overview_source)
        self.assertNotIn("self.system(", overview_source)

        from src.control_plane import server
        get_source = inspect.getsource(server.ControlPlaneHandler._api_get)
        health_source = inspect.getsource(server.ControlPlaneServer.health)
        self.assertNotIn("sync_recent", get_source)
        self.assertNotIn("read_recent_log", inspect.getsource(server))
        self.assertNotIn("integrity_check", health_source)

    def test_required_indexes_exist(self) -> None:
        indexes = {
            row[1]
            for row in self.store.connection.execute(
                "SELECT type,name FROM sqlite_master WHERE type='index'"
            )
        }
        self.assertIn("analyses_request_created_idx", indexes)
        self.assertIn("jobs_request_id_idx", indexes)
        self.assertIn("annotations_request_origin_created_idx", indexes)


class DurableMaintenanceTests(unittest.TestCase):
    def test_jobs_retry_and_complete_durably(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = AnnotationStore(root / "annotations.sqlite3")
            migrate(store.connection)
            attempts = {"count": 0}

            def mirror(_path):
                attempts["count"] += 1
                if attempts["count"] < 3:
                    raise OSError("temporary failure")
                return 1

            store.mirror_audit_events = mirror
            store.export_approved = lambda *_args: {"output": str(root / "approved.jsonl")}
            store.backup = lambda *_args: root / "backup.sqlite3"
            worker = MaintenanceWorker(
                store,
                approved_export=root / "approved.jsonl",
                audit_mirror=root / "events.jsonl",
                backup_dir=root / "backups",
                locked_eval=root / "locked.jsonl",
            )
            worker.start()
            worker.schedule(42)
            deadline = time.monotonic() + 5
            state = ""
            while time.monotonic() < deadline:
                row = store.connection.execute(
                    "SELECT state,attempts FROM maintenance_jobs "
                    "WHERE dedupe_key='decision:42'"
                ).fetchone()
                state = row["state"]
                if state == "completed":
                    break
                time.sleep(0.05)
            worker.stop()

            self.assertEqual(state, "completed")
            self.assertEqual(row["attempts"], 3)
            self.assertEqual(attempts["count"], 3)
            store.close()


class HttpReadPathTests(unittest.TestCase):
    def test_react_shell_and_reads_are_fast_and_do_not_mutate_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = AnnotationStore(root / "annotations.sqlite3")
            migrate(store.connection)
            store.import_proxy_record(proxy_record())
            fake_worker = SimpleNamespace(threads=[], queue=queue.Queue())
            args = SimpleNamespace(
                mlflow_tracking_uri="http://127.0.0.1:9",
                log_file=root / "missing-proxy.jsonl",
                approved_export=root / "approved.jsonl",
                audit_mirror=root / "events.jsonl",
                backup_dir=root / "backups",
                locked_eval=root / "locked.jsonl",
                luna_model="gpt-5.6-luna",
                luna_reasoning_effort="xhigh",
            )
            server = ControlPlaneServer(("127.0.0.1", 0), store, fake_worker, args)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = "http://127.0.0.1:" + str(server.server_address[1])
            try:
                before = {
                    table: store.connection.execute(
                        "SELECT COUNT(*) FROM " + table
                    ).fetchone()[0]
                    for table in ("samples", "annotations", "decisions", "audit_events")
                }
                timings = {}
                bodies = {}
                for route in ("/", "/api/v1/overview", "/samples/request-1"):
                    started = time.perf_counter()
                    with urlopen(base + route, timeout=2) as response:
                        bodies[route] = response.read()
                    timings[route] = time.perf_counter() - started
                after = {
                    table: store.connection.execute(
                        "SELECT COUNT(*) FROM " + table
                    ).fetchone()[0]
                    for table in before
                }

                self.assertIn(b'<div id="root"></div>', bodies["/"])
                self.assertIn(b'"stats"', bodies["/api/v1/overview"])
                self.assertIn(b'<div id="root"></div>', bodies["/samples/request-1"])
                self.assertEqual(after, before)
                self.assertLess(timings["/"], 0.05)
                self.assertLess(timings["/api/v1/overview"], 0.15)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                store.close()


if __name__ == "__main__":
    unittest.main()
