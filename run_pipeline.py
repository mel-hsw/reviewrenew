#!/usr/bin/env python3
"""Run the agentic pipeline: RAG index → LLM analysis → (optional) two image models.

Use ``--skip-images`` for chunking strategy experiments without image API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.agent import run_pipeline
from src.config import PRODUCTS_PATH
from src.models import ProductCatalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Review → RAG → image generation pipeline")
    parser.add_argument(
        "--data",
        type=Path,
        default=PRODUCTS_PATH,
        help=f"Path to products JSON (default: {PRODUCTS_PATH})",
    )
    parser.add_argument(
        "--images-per-model",
        type=int,
        default=3,
        help="Number of distinct prompts / images to generate per image model (3–5 typical).",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Run RAG indexing + LLM bundles only (no image API calls). Useful for chunking A/B.",
    )
    args = parser.parse_args()
    if not args.data.exists():
        print(
            f"Missing {args.data}. Copy data/products.example.json to data/products.json and fill it.",
            file=sys.stderr,
        )
        return 1
    raw = json.loads(args.data.read_text(encoding="utf-8"))
    catalog = ProductCatalog.model_validate(raw)
    if not catalog.products:
        print("No products in catalog.", file=sys.stderr)
        return 1
    state = run_pipeline(
        catalog.products,
        images_per_model=args.images_per_model,
        skip_images=args.skip_images,
    )
    print(f"Done. {len(state.logs)} steps. Outputs under outputs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
