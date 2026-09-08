#!/usr/bin/env python3
"""Enqueue fresh Luna analysis for every pending member of a bootstrap cohort."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from labeling.annotation_store import DEFAULT_DATABASE


def enqueue(base_url: str, request_id: str) -> tuple[str, str]:
    request = Request(
        f"{base_url.rstrip('/')}/api/samples/{quote(request_id, safe='')}/analyze",
        data=b"{}", method="POST", headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read())
        return request_id, str(result.get("state", "queued"))
    except (HTTPError, URLError, TimeoutError) as exc:
        return request_id, f"error: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--server", default="http://127.0.0.1:8003")
    parser.add_argument("--parallel", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    connection = sqlite3.connect(args.database)
    row = connection.execute("SELECT id FROM canonical_cohorts WHERE name=?", (args.cohort,)).fetchone()
    if not row:
        raise SystemExit(f"Unknown cohort: {args.cohort}")
    ids = [item[0] for item in connection.execute(
        "SELECT request_id FROM canonical_cohort_members WHERE cohort_id=? ORDER BY request_id", row
    )]
    failures = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(enqueue, args.server, request_id) for request_id in ids]
        for index, future in enumerate(as_completed(futures), 1):
            request_id, state = future.result()
            if state.startswith("error:"):
                failures.append((request_id, state))
            if index % 50 == 0 or index == len(ids):
                print(f"[{index}/{len(ids)}] queued; failures={len(failures)}")
    if failures:
        print(json.dumps({"failures": failures}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
