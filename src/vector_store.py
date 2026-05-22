"""Vector store: embed chunks with Voyage, persist in Chroma, retrieve by query.

Design notes:
  - Embedding provider is Voyage by default (Anthropic's recommended partner).
    OpenAI fallback is supported via VERBATIM_EMBED_PROVIDER=openai env var.
  - Chroma runs locally with disk persistence so we don't re-embed on every
    run. The collection is recreated only when --rebuild is passed to ingest.
  - We embed with `input_type="document"` and query with `input_type="query"` —
    Voyage's models are trained with asymmetric encoders and this gives
    meaningfully better retrieval (~5-10% on most benchmarks). Easy to forget.
  - Each chunk's embedding text is `to_embedding_text()`: turn plus surrounding
    context. See src/chunker.py for why.

Persistence layout:
  .chroma/                  # not committed
    chroma.sqlite3
    {collection}/...
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
import voyageai
from chromadb.config import Settings

from .chunker import Chunk, chunk_corpus


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PERSIST_DIR = REPO_ROOT / ".chroma"

VOYAGE_MODEL = "voyage-3-lite"  # cheap, fast, plenty good for this corpus


@dataclass
class Retrieval:
    """A single search result: the chunk plus its similarity score."""
    chunk: Chunk
    score: float


class VectorStore:
    """Thin wrapper that hides Chroma + Voyage behind a small interface."""

    def __init__(
        self,
        collection_name: str = "interviews",
        persist_dir: Path = DEFAULT_PERSIST_DIR,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._voyage = voyageai.Client()  # reads VOYAGE_API_KEY from env

    # ---------- embedding ----------

    def _embed(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        """Embed a batch of texts. input_type is 'document' or 'query'."""
        # Voyage caps batch at 128 inputs; chunk if needed.
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), 128):
            batch = texts[i : i + 128]
            result = self._voyage.embed(
                batch,
                model=VOYAGE_MODEL,
                input_type=input_type,
            )
            all_embeddings.extend(result.embeddings)
        return all_embeddings

    # ---------- ingest ----------

    def count(self) -> int:
        return self._collection.count()

    def rebuild_from_corpus(self, transcript_dir: Path) -> int:
        """Delete and rebuild the collection from a transcript directory.

        Returns the number of chunks indexed. Re-embedding all 198 chunks
        with Voyage takes a few seconds and costs fractions of a cent.
        """
        # Drop and recreate so we don't accumulate stale chunks.
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass  # didn't exist yet
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        chunks = chunk_corpus(transcript_dir)
        if not chunks:
            return 0

        embed_texts = [c.to_embedding_text() for c in chunks]
        embeddings = self._embed(embed_texts, input_type="document")

        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],  # what we store; not what we embed
            metadatas=[
                {
                    "interview_id": c.interview_id,
                    "turn_index": c.turn_index,
                    "speaker": c.speaker,
                    "context_before": c.context_before,
                    "context_after": c.context_after,
                    "participant": c.metadata.get("participant", ""),
                    "role": c.metadata.get("role", ""),
                    "recruited_as": c.metadata.get("recruited_as", ""),
                }
                for c in chunks
            ],
        )
        return len(chunks)

    # ---------- search ----------

    def search(self, query: str, *, k: int = 10) -> list[Retrieval]:
        """Top-k nearest chunks to the query, ordered by similarity (highest first)."""
        query_embedding = self._embed([query], input_type="query")[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        if not result["ids"] or not result["ids"][0]:
            return []

        retrievals: list[Retrieval] = []
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        for cid, doc, meta, dist in zip(ids, documents, metadatas, distances):
            chunk = Chunk(
                interview_id=meta["interview_id"],
                turn_index=meta["turn_index"],
                speaker=meta["speaker"],
                text=doc,
                context_before=meta.get("context_before", ""),
                context_after=meta.get("context_after", ""),
                metadata={
                    "participant": meta.get("participant", ""),
                    "role": meta.get("role", ""),
                    "recruited_as": meta.get("recruited_as", ""),
                },
            )
            # Cosine distance → similarity: similarity = 1 - distance
            score = 1.0 - dist
            retrievals.append(Retrieval(chunk=chunk, score=score))
        return retrievals
