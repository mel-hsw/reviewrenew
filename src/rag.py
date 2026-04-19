from __future__ import annotations

import uuid

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from . import config
from .chunking import product_to_chunks
from .models import Product, TextChunk


def _client() -> OpenAI:
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _embed(texts: list[str]) -> list[list[float]]:
    resp = _client().embeddings.create(model=config.OPENAI_EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


class ReviewRAGIndex:
    """Per-product Chroma collection: embed chunks and retrieve by semantic similarity."""

    def __init__(self, persist_dir: str | None = None) -> None:
        path = persist_dir or str(config.CHROMA_DIR)
        self._chroma = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )

    def reset_collection(self, name: str) -> None:
        try:
            self._chroma.delete_collection(name)
        except Exception:
            pass

    def index_product(self, product: Product) -> int:
        chunks = product_to_chunks(
            product,
            strategy=config.CHUNK_STRATEGY,
            max_tokens=config.CHUNK_MAX_TOKENS,
            overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
        )
        if not chunks:
            return 0
        coll_name = _collection_name(product.asin)
        self.reset_collection(coll_name)
        coll = self._chroma.get_or_create_collection(name=coll_name, metadata={"asin": product.asin})
        texts = [c.text for c in chunks]
        embs = _embed(texts)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = []
        for c in chunks:
            md: dict[str, str | int] = {
                "asin": c.asin,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "review_index": c.review_index if c.review_index is not None else -1,
            }
            metadatas.append(md)
        coll.add(ids=ids, embeddings=embs, documents=texts, metadatas=metadatas)
        return len(chunks)

    def retrieve(
        self,
        asin: str,
        query: str,
        k: int = 8,
    ) -> list[TextChunk]:
        coll_name = _collection_name(asin)
        coll = self._chroma.get_collection(name=coll_name)
        qemb = _embed([query])[0]
        res = coll.query(
            query_embeddings=[qemb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        out: list[TextChunk] = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        for doc, meta in zip(docs, metas):
            if not doc or not meta:
                continue
            ri = int(meta.get("review_index", -1))
            out.append(
                TextChunk(
                    asin=str(meta.get("asin", asin)),
                    source=meta.get("source", "review"),  # type: ignore[arg-type]
                    review_index=ri if ri >= 0 else None,
                    text=doc,
                    chunk_index=int(meta.get("chunk_index", 0)),
                )
            )
        return out


def _collection_name(asin: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in asin)
    return f"product_{safe}"
