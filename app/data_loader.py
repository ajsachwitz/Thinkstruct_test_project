from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatentChunk:
    doc_number: str
    source_type: str  # "abstract" or "claim"
    text: str


@dataclass(frozen=True)
class PatentDocument:
    doc_number: str
    title: str
    abstract: str


def load_patent_dataset(
    data_dir: Path,
) -> Tuple[List[PatentChunk], Dict[str, PatentDocument]]:
    """
    Load patent abstracts and claims from all JSON files in the directory.
    Each abstract and each claim becomes its own searchable chunk.
    """
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(
            f"Patent data directory '{data_dir}' does not exist or is not a directory."
        )

    chunks: List[PatentChunk] = []
    documents: Dict[str, PatentDocument] = {}
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in patent data directory '{data_dir}'."
        )

    for path in json_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping %s due to JSON error: %s", path.name, exc)
            continue

        if not isinstance(payload, list):
            logger.warning("Skipping %s because it does not contain a list.", path.name)
            continue

        for entry in payload:
            doc_number = str(entry.get("doc_number") or entry.get("number") or "").strip()
            if not doc_number:
                continue

            title = _clean_text(entry.get("title"))
            abstract = _clean_text(entry.get("abstract"))
            if doc_number not in documents:
                documents[doc_number] = PatentDocument(
                    doc_number=doc_number,
                    title=title,
                    abstract=abstract,
                )

            if abstract:
                chunks.append(
                    PatentChunk(doc_number=doc_number, source_type="abstract", text=abstract)
                )

            claims = entry.get("claims") or []
            if isinstance(claims, Iterable):
                for claim in claims:
                    claim_text = _clean_text(claim)
                    if claim_text:
                        chunks.append(
                            PatentChunk(doc_number=doc_number, source_type="claim", text=claim_text)
                        )

    logger.info(
        "Loaded %s searchable chunks (%s documents) from %s files.",
        len(chunks),
        len(documents),
        len(json_files),
    )
    return chunks, documents


def load_patent_chunks(data_dir: Path) -> List[PatentChunk]:
    chunks, _ = load_patent_dataset(data_dir)
    return chunks


def _clean_text(value: object) -> str:
    if not value or not isinstance(value, str):
        return ""
    return " ".join(value.split())
