"""Ingest the transcript corpus into the local vector store.

Run once before any pipeline that needs retrieval:

    python -m src.ingest                   # uses transcripts/
    python -m src.ingest --dir job_transcripts/   # different corpus
    python -m src.ingest --collection jobs        # named collection

You need VOYAGE_API_KEY set in your environment (or .env file).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from .vector_store import VectorStore


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")

    if not os.getenv("VOYAGE_API_KEY"):
        sys.exit(
            "VOYAGE_API_KEY not set. Put it in .env or export it.\n"
            "Get a key at https://www.voyageai.com (free tier covers this project easily)."
        )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default=str(REPO_ROOT / "transcripts"),
        help="directory of transcript .md files",
    )
    parser.add_argument(
        "--collection",
        default="interviews",
        help="vector store collection name (use a different name for a different corpus)",
    )
    args = parser.parse_args()

    transcript_dir = Path(args.dir)
    if not transcript_dir.exists():
        sys.exit(f"transcript dir not found: {transcript_dir}")

    print(f"Ingesting from: {transcript_dir}")
    print(f"Collection:     {args.collection}")
    t0 = time.perf_counter()
    store = VectorStore(collection_name=args.collection)
    n = store.rebuild_from_corpus(transcript_dir)
    elapsed = time.perf_counter() - t0
    print(f"Indexed {n} chunks in {elapsed:.1f}s.")
    print(f"Vector store persisted at: {store.persist_dir}")


if __name__ == "__main__":
    main()
