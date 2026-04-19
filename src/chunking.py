from __future__ import annotations

import tiktoken

from .models import Product, Review, TextChunk


def _encoding():
    return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def chunk_text(
    text: str,
    max_tokens: int = 400,
    overlap_tokens: int = 60,
) -> list[str]:
    enc = _encoding()
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return [enc.decode(ids)]
    out: list[str] = []
    start = 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        out.append(enc.decode(ids[start:end]))
        if end >= len(ids):
            break
        start = end - overlap_tokens
        if start < 0:
            start = 0
    return out


def chunk_paragraph_batches(text: str, max_tokens: int) -> list[str]:
    """
    Pack paragraphs (split on blank lines) into chunks without exceeding max_tokens.
    If a single paragraph exceeds max_tokens, fall back to sliding chunk_text on it.
    """
    text = text.strip()
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return []
    out: list[str] = []
    current: list[str] = []
    current_tok = 0
    for p in paras:
        pt = _count_tokens(p)
        if pt > max_tokens:
            if current:
                out.append("\n\n".join(current))
                current = []
                current_tok = 0
            out.extend(chunk_text(p, max_tokens=max_tokens, overlap_tokens=0))
            continue
        add_tok = pt + (2 if current else 0)
        if current and current_tok + add_tok > max_tokens:
            out.append("\n\n".join(current))
            current = [p]
            current_tok = pt
        else:
            current.append(p)
            current_tok += add_tok
    if current:
        out.append("\n\n".join(current))
    return out


def product_to_chunks(
    product: Product,
    *,
    strategy: str = "sliding",
    max_tokens: int = 400,
    overlap_tokens: int = 60,
) -> list[TextChunk]:
    """
    strategy:
      - sliding: fixed-size token windows with overlap (description and each review body).
      - paragraph_batch: merge whole paragraphs up to max_tokens per chunk (preserves bullet blocks).
    """
    strategy = (strategy or "sliding").lower().strip()
    chunks: list[TextChunk] = []
    desc = product.description.strip()
    if desc:
        if strategy == "paragraph_batch":
            parts = chunk_paragraph_batches(desc, max_tokens)
        else:
            parts = chunk_text(desc, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        for i, part in enumerate(parts):
            chunks.append(
                TextChunk(
                    asin=product.asin,
                    source="description",
                    review_index=None,
                    text=part,
                    chunk_index=i,
                )
            )
    for ri, rev in enumerate(product.reviews):
        body = _review_body(rev)
        if not body:
            continue
        if strategy == "paragraph_batch":
            parts = chunk_paragraph_batches(body, max_tokens)
        else:
            parts = chunk_text(body, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        for i, part in enumerate(parts):
            chunks.append(
                TextChunk(
                    asin=product.asin,
                    source="review",
                    review_index=ri,
                    text=part,
                    chunk_index=i,
                )
            )
    return chunks


def _review_body(rev: Review) -> str:
    parts: list[str] = []
    if rev.title:
        parts.append(rev.title.strip())
    parts.append(rev.text.strip())
    return "\n\n".join(p for p in parts if p)
