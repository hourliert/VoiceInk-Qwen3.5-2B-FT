#!/usr/bin/env python3
"""Show dataset distribution by input word count.

Usage:
    python3 src/training/show_distribution.py
    python3 src/training/show_distribution.py --extra-input datasets/synthetic/labeled.jsonl
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common.extract import extract_from_record

BUCKETS = [
    (0, 50, "0-50 words"),
    (50, 100, "50-100 words"),
    (100, 200, "100-200 words"),
    (200, 500, "200-500 words"),
    (500, float("inf"), "500+ words"),
]


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def word_count(record):
    components = extract_from_record(record)
    return len(components["transcript"].split())


def bucket_counts(records):
    counts = {label: 0 for _, _, label in BUCKETS}
    for r in records:
        wc = word_count(r)
        for lo, hi, label in BUCKETS:
            if lo <= wc < hi:
                counts[label] += 1
                break
    return counts


def print_table(columns):
    """Print a distribution table with one or more columns.

    columns: list of (header, counts_dict, total) tuples
    """
    label_width = max(len(label) for _, _, label in BUCKETS)
    col_width = max(max(len(h) for h, _, _ in columns), 18)

    # Header
    header = f"{'Input length':<{label_width}}"
    for col_header, _, _ in columns:
        header += f"  {col_header:>{col_width}}"
    print(header)
    print("-" * len(header))

    # Rows
    for _, _, label in BUCKETS:
        row = f"{label:<{label_width}}"
        for _, counts, total in columns:
            n = counts[label]
            pct = n / total * 100 if total else 0
            cell = f"{n:,} ({pct:.1f}%)"
            row += f"  {cell:>{col_width}}"
        print(row)

    # Total
    row = f"{'Total':<{label_width}}"
    for _, _, total in columns:
        cell = f"{total:,}"
        row += f"  {cell:>{col_width}}"
    print("-" * len(header))
    print(row)


def main():
    parser = argparse.ArgumentParser(description="Show dataset distribution by input word count")
    parser.add_argument("--input", default="datasets/labeled.jsonl", help="Primary labeled data")
    parser.add_argument("--extra-input", help="Additional data (e.g. synthetic) to show side-by-side")
    args = parser.parse_args()

    real = load_records(args.input)
    real_counts = bucket_counts(real)

    if args.extra_input:
        extra = load_records(args.extra_input)
        combined = real + extra
        combined_counts = bucket_counts(combined)
        print_table([
            (f"Real only ({len(real)})", real_counts, len(real)),
            (f"+ synthetic ({len(combined)})", combined_counts, len(combined)),
        ])
    else:
        print_table([
            (f"All samples ({len(real)})", real_counts, len(real)),
        ])


if __name__ == "__main__":
    main()
