#!/usr/bin/env python3
"""
Stream Amazon Reviews '23 *metadata* JSONL and collect items with rating_number > N.

Docs: https://amazon-reviews-2023.github.io/#quick-start

Requires files like meta_Pet_Supplies.jsonl (meta download), not Pet_Supplies.jsonl (reviews).
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path


def summarize_row(row: dict) -> dict:
    """Fields that are easy to review in a report."""
    return {
        "parent_asin": row.get("parent_asin"),
        "title": row.get("title"),
        "rating_number": row.get("rating_number"),
        "average_rating": row.get("average_rating"),
        "main_category": row.get("main_category"),
        "store": row.get("store"),
        "price": row.get("price"),
    }


def stream_filter(
    path: Path,
    min_ratings: int,
    limit: int,
    sort_by: str,
) -> list[dict]:
    """
    min_ratings: keep rows with rating_number > min_ratings (strict).
    For sort rating_number_desc, use a size-`limit` min-heap so one pass works on huge files.
    """
    if sort_by == "rating_number_desc":
        heap: list[tuple[int, int, dict]] = []
        counter = 0
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rn = row.get("rating_number")
                if rn is None:
                    continue
                rn_i = int(rn)
                if rn_i <= min_ratings:
                    continue
                counter += 1
                item = (rn_i, counter, row)
                if len(heap) < limit:
                    heapq.heappush(heap, item)
                elif rn_i > heap[0][0]:
                    heapq.heapreplace(heap, item)
        rows = [t[2] for t in sorted(heap, key=lambda x: (-x[0], x[1]))]
        return rows

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rn = row.get("rating_number")
            if rn is None or int(rn) <= min_ratings:
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter meta JSONL by rating_number (Amazon Reviews 2023)."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Paths to meta_*.jsonl files (e.g. data/raw/amazon_reviews_2023/meta_Pet_Supplies.jsonl)",
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=1000,
        help="Keep items with rating_number strictly greater than this value (default: 1000).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Max items to output per file (default: 50).",
    )
    parser.add_argument(
        "--sort",
        choices=("rating_number_desc", "none"),
        default="rating_number_desc",
        help="Order before taking first --count (default: highest rating_number first).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Emit full metadata objects instead of a short summary dict.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write combined JSON to this file (UTF-8).",
    )
    args = parser.parse_args()

    if not args.files:
        print(
            "Usage: python scripts/filter_meta_by_rating.py data/raw/amazon_reviews_2023/meta_Pet_Supplies.jsonl ...\n"
            "Download meta_*.jsonl from https://amazon-reviews-2023.github.io/#quick-start",
            file=sys.stderr,
        )
        return 1

    report: dict[str, list] = {}
    for path in args.files:
        if not path.exists():
            print(f"Missing file: {path}", file=sys.stderr)
            return 1
        rows = stream_filter(path, args.min_ratings, args.count, args.sort)
        key = path.name
        if args.full:
            report[key] = rows
        else:
            report[key] = [summarize_row(r) for r in rows]

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
