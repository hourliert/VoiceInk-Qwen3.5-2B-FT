#!/usr/bin/env python3
"""Create a consistent online backup of the local MLflow SQLite store."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "training/mlflow/mlflow.db")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "training/backups/mlflow")
    parser.add_argument("--retain", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        print(f"MLflow database does not exist: {args.source}")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = args.output_dir / f"mlflow-{stamp}.sqlite3"
    source = sqlite3.connect(f"file:{args.source.resolve()}?mode=ro", uri=True)
    target = sqlite3.connect(destination)
    try:
        source.backup(target)
        if target.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("MLflow backup integrity check failed")
    finally:
        target.close()
        source.close()
    backups = sorted(args.output_dir.glob("mlflow-*.sqlite3"), reverse=True)
    for stale in backups[args.retain:]:
        stale.unlink()
    print(destination)


if __name__ == "__main__":
    main()
