#!/usr/bin/env python3
"""
Build data/products.json + data/products_manifest.json from Amazon Reviews '23
meta + review JSONL for a fixed list of parent ASINs.

Raw slices live under data/raw/amazon_reviews_2023/ (see data/README.md).

Edit SELECTED below or pass --config path.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "raw" / "amazon_reviews_2023"

# (key, parent_asin, meta_filename, reviews_filename)
SELECTED: list[tuple[str, str, str, str]] = [
    (
        "pet_bed",
        "B0C4BFWLNC",
        "meta_Pet_Supplies.jsonl",
        "Pet_Supplies.jsonl",
    ),
    (
        "cerave_hydrating_facial_cleanser",
        "B0BG9Q18ZZ",
        "meta_Beauty_and_Personal_Care.jsonl",
        "Beauty_and_Personal_Care.jsonl",
    ),
    (
        "hat_washer",
        "B0BXSQN4HV",
        "meta_Appliances.jsonl",
        "Appliances.jsonl",
    ),
]


def _ts_to_date(ms: int | float | None) -> str | None:
    if ms is None:
        return None
    try:
        sec = float(ms) / 1000.0
        dt = datetime.fromtimestamp(sec, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return None


def load_meta_row(path: Path, parent_asin: str) -> dict | None:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("parent_asin") == parent_asin:
                return row
    return None


def meta_to_description(row: dict) -> str:
    parts: list[str] = []
    feats = row.get("features") or []
    if feats:
        parts.append("Features:\n" + "\n".join(f"- {f}" for f in feats))
    desc = row.get("description")
    if desc:
        if isinstance(desc, list):
            parts.append("\n".join(str(x) for x in desc))
        else:
            parts.append(str(desc))
    det = row.get("details") or {}
    if det:
        parts.append("Product details:\n" + "\n".join(f"{k}: {v}" for k, v in det.items()))
    cats = row.get("categories") or []
    if cats:
        parts.append("Categories: " + " > ".join(str(c) for c in cats))
    return "\n\n".join(parts)


def meta_to_image_urls(row: dict, limit: int = 6) -> list[str]:
    out: list[str] = []
    for im in row.get("images") or []:
        if not isinstance(im, dict):
            continue
        url = im.get("large") or im.get("hi_res") or im.get("thumb")
        if url:
            out.append(url)
        if len(out) >= limit:
            break
    return out


def collect_reviews(
    path: Path,
    parent_asin: str,
    *,
    min_text_chars: int,
    target_reviews: int,
) -> tuple[list[dict], dict]:
    """
    Collect reviews for one parent_asin.

    - Drops empty bodies and reviews with len(text) <= min_text_chars (default: 20
      characters means keep strictly *more than* 20, so min_text_chars=21).
    - Sorts by time (newest first) when timestamps exist.
    - Keeps up to `target_reviews` rows (default 500) for semantically rich coverage.

    Returns (reviews, stats dict).
    """
    found: list[dict] = []
    total_rows = 0
    with path.open("rb") as fp:
        for raw in fp:
            d = json.loads(raw)
            pa = d.get("parent_asin") or d.get("asin")
            if pa != parent_asin:
                continue
            total_rows += 1
            ts = d.get("timestamp") or d.get("sort_timestamp")
            rating = d.get("rating")
            if rating is not None:
                try:
                    rating_i = int(round(float(rating)))
                except (TypeError, ValueError):
                    rating_i = None
            else:
                rating_i = None
            text = (d.get("text") or "").strip()
            if len(text) <= min_text_chars:
                continue
            found.append(
                {
                    "rating": rating_i,
                    "title": (d.get("title") or "").strip() or None,
                    "text": text,
                    "date": _ts_to_date(ts),
                    "_ts": int(ts) if ts is not None else 0,
                }
            )

    found.sort(key=lambda r: r["_ts"], reverse=True)
    for r in found:
        del r["_ts"]

    qualifying = len(found)
    capped = found[:target_reviews]
    stats = {
        "review_rows_in_file_for_asin": total_rows,
        "after_length_filter": qualifying,
        "min_text_chars_exclusive": min_text_chars,
        "rule": f"len(text) > {min_text_chars} (strictly more than {min_text_chars} characters).",
        "target_reviews": target_reviews,
        "included": len(capped),
        "reached_target_count": qualifying >= target_reviews,
    }
    if qualifying < target_reviews:
        stats["warning"] = (
            f"Only {qualifying} reviews pass the length filter; fewer than target {target_reviews}."
        )
    return capped, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-reviews",
        type=int,
        default=500,
        help="Max reviews per product after cleaning (default 500).",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=20,
        help="Include only reviews with more than this many characters (default 20 => strictly longer than 20).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "products.json",
        help="Output catalog JSON.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "products_manifest.json",
    )
    args = parser.parse_args()

    products: list[dict] = []
    manifest: dict = {"items": []}

    for key, asin, meta_name, rev_name in SELECTED:
        meta_path = DB / meta_name
        rev_path = DB / rev_name
        if not meta_path.exists():
            raise SystemExit(f"Missing {meta_path}")
        if not rev_path.exists():
            raise SystemExit(f"Missing {rev_path}")

        meta = load_meta_row(meta_path, asin)
        if not meta:
            raise SystemExit(f"parent_asin {asin} not found in {meta_name}")

        reviews, rev_stats = collect_reviews(
            rev_path,
            asin,
            min_text_chars=args.min_text_chars,
            target_reviews=args.target_reviews,
        )

        manifest["items"].append(
            {
                "key": key,
                "parent_asin": asin,
                "meta_file": meta_name,
                "reviews_file": rev_name,
                "review_cleaning": rev_stats,
            }
        )

        products.append(
            {
                "asin": asin,
                "title": meta.get("title") or "",
                "category": meta.get("main_category") or "",
                "description": meta_to_description(meta),
                "listing_image_urls": meta_to_image_urls(meta),
                "reviews": reviews,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"products": products}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
