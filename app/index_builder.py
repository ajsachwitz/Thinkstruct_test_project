from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import faiss

from app.config import settings
from app.data_loader import PatentChunk, PatentDocument, load_patent_dataset
from app.embeddings import embed_texts, normalize_embeddings

logger = logging.getLogger(__name__)


def ensure_index_artifacts(force_rebuild: bool = False, chunk_limit: int | None = None) -> None:
    """
    Create FAISS + metadata artifacts if they are missing or force flag is set.
    """
    artifacts_exist = (
        settings.abstract_index_path.exists()
        and settings.abstract_metadata_path.exists()
        and settings.claims_index_path.exists()
        and settings.claims_metadata_path.exists()
        and settings.doc_metadata_path.exists()
    )

    if not force_rebuild and artifacts_exist:
        return

    logger.info("Building FAISS indices (force=%s)...", force_rebuild)
    build_index_artifacts(chunk_limit=chunk_limit)


def build_index_artifacts(chunk_limit: int | None = None) -> None:
    chunks, documents = load_patent_dataset(settings.data_dir)
    if chunk_limit:
        chunks = chunks[:chunk_limit]
        doc_whitelist = {chunk.doc_number for chunk in chunks}
        documents = {doc_number: documents[doc_number] for doc_number in doc_whitelist if doc_number in documents}

    if not chunks:
        raise RuntimeError("No patent chunks available to build the index.")

    abstract_chunks = [chunk for chunk in chunks if chunk.source_type == "abstract"]
    claim_chunks = [chunk for chunk in chunks if chunk.source_type == "claim"]

    if not abstract_chunks:
        raise RuntimeError("No abstract chunks found; cannot build abstract index.")
    if not claim_chunks:
        raise RuntimeError("No claim chunks found; cannot build claim index.")

    logger.info("Embedding %s abstract chunks...", len(abstract_chunks))
    build_index_for_chunks(
        abstract_chunks,
        settings.abstract_index_path,
        settings.abstract_metadata_path,
        label="abstract",
    )

    logger.info("Embedding %s claim chunks...", len(claim_chunks))
    build_index_for_chunks(
        claim_chunks,
        settings.claims_index_path,
        settings.claims_metadata_path,
        label="claim",
    )

    _write_doc_metadata(documents)


def build_index_for_chunks(
    chunks: list[PatentChunk],
    index_path: Path,
    metadata_path: Path,
    label: str,
) -> None:
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(
        texts,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=True,
    )
    embeddings = normalize_embeddings(embeddings)

    dim = embeddings.shape[1]
    logger.info("Creating %s FAISS IndexFlatIP (dim=%s)", label, dim)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info("%s index written to %s", label.capitalize(), index_path)

    metadata_payload = [
        {
            "doc_number": chunk.doc_number,
            "source_type": chunk.source_type,
            "source_text": chunk.text,
        }
        for chunk in chunks
    ]
    metadata_path.write_text(
        json.dumps(metadata_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("%s metadata written to %s", label.capitalize(), metadata_path)


def _write_doc_metadata(documents: dict[str, PatentDocument]) -> None:
    payload = {
        doc_number: {
            "title": doc.title,
            "abstract": doc.abstract,
        }
        for doc_number, doc in documents.items()
    }
    settings.doc_metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Document metadata written to %s", settings.doc_metadata_path)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or rebuild the semantic patent FAISS index."
    )
    parser.add_argument("--force", action="store_true", help="Force rebuild artifacts.")
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=None,
        help="Limit number of chunks for quicker test builds.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    ensure_index_artifacts(force_rebuild=args.force, chunk_limit=args.chunk_limit)


if __name__ == "__main__":
    main()
