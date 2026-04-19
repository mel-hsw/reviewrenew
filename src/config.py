from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")


def _req(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


OPENAI_API_KEY = _req("OPENAI_API_KEY")
OPENAI_TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4o").strip()
# Optional: separate text model for Q3 creative prompts (defaults to OPENAI_TEXT_MODEL).
OPENAI_Q3_TEXT_MODEL = (os.environ.get("OPENAI_Q3_TEXT_MODEL", "").strip() or OPENAI_TEXT_MODEL)
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
# Image generation: one OpenAI Images model + one Gemini native image model (Q3).
# OPENAI_IMAGE_MODEL is the canonical name; OPENAI_IMAGE_MODEL_PRIMARY is a legacy alias.
OPENAI_IMAGE_MODEL = (
    os.environ.get("OPENAI_IMAGE_MODEL", "").strip()
    or os.environ.get("OPENAI_IMAGE_MODEL_PRIMARY", "chatgpt-image-latest").strip()
)


def _gemini_image_model_from_env() -> str:
    """
    GEMINI_IMAGE_MODEL: explicit Gemini image model id, or empty to skip Gemini runs.
    If unset: use legacy OPENAI_IMAGE_MODEL_SECONDARY when it is a gemini-* id; if SECONDARY
    is set to a non-Gemini model (legacy second OpenAI), leave Gemini disabled; if SECONDARY is
    empty too, default to gemini-3.1-flash-image-preview (typical Q3 two-provider setup).
    """
    if "GEMINI_IMAGE_MODEL" in os.environ:
        return os.environ["GEMINI_IMAGE_MODEL"].strip()
    sec = os.environ.get("OPENAI_IMAGE_MODEL_SECONDARY", "").strip()
    if sec.lower().startswith("gemini-"):
        return sec
    if sec:
        return ""
    return "gemini-3.1-flash-image-preview"


GEMINI_IMAGE_MODEL = _gemini_image_model_from_env()
# Image API: size is shared; quality semantics depend on model (see OpenAI images guide).
OPENAI_IMAGE_SIZE = os.environ.get("OPENAI_IMAGE_SIZE", "1024x1024").strip()
# Optional. For GPT image models: low|medium|high|auto. For dall-e-3: standard|hd. Empty = omit.
OPENAI_IMAGE_QUALITY = os.environ.get("OPENAI_IMAGE_QUALITY", "").strip()

# Gemini (Nano Banana) native image generation — uses GEMINI_IMAGE_MODEL + GEMINI_API_KEY.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
# Output resolution for Gemini 3.x image models: 512 | 1K | 2K | 4K (uppercase K; ignored for gemini-2.5-flash-image).
GEMINI_IMAGE_RESOLUTION = os.environ.get("GEMINI_IMAGE_RESOLUTION", "1K").strip()

# Optional: append to RAG bundle shot prompts per provider (strong shared base + light tuning).
OPENAI_IMAGE_PROMPT_SUFFIX = os.environ.get("OPENAI_IMAGE_PROMPT_SUFFIX", "").strip()
GEMINI_IMAGE_PROMPT_SUFFIX = os.environ.get("GEMINI_IMAGE_PROMPT_SUFFIX", "").strip()

DATA_DIR = _ROOT / "data"
PRODUCTS_PATH = DATA_DIR / "products.json"
CHROMA_DIR = DATA_DIR / "chroma"
OUTPUTS_DIR = _ROOT / "outputs"

# RAG chunking (Q2): strategy + tokenizer limits — see docs/report_draft.md §2.2 and src/chunking.py
# sliding: fixed-size windows with overlap (good for long reviews).
# paragraph_batch: pack whole paragraphs (split on \\n\\n) into batches up to max_tokens (good for bullets/features).
# Default paragraph_batch: short reviews dominate typical Amazon corpora after cleaning; override via .env.
_raw_chunk = os.environ.get("CHUNK_STRATEGY", "paragraph_batch").strip().lower()
CHUNK_STRATEGY = _raw_chunk if _raw_chunk in ("sliding", "paragraph_batch") else "paragraph_batch"
CHUNK_MAX_TOKENS = int(os.environ.get("CHUNK_MAX_TOKENS", "400"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "60"))
