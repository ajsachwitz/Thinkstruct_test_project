from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from threading import Lock
from typing import List, Sequence

import faiss
import numpy as np

from app.config import settings
from app.embeddings import embed_texts, normalize_embeddings
from app.index_builder import ensure_index_artifacts
from app.scraper import PatentScrapeResult, scrape_patent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    doc_number: str
    source_type: str
    source_text: str
    score: float | None = None
    query_claim_text: str | None = None


@dataclass(frozen=True)
class SearchResponse:
    patent_id: str
    scraped_title: str
    scraped_abstract: str
    scraped_claims: Sequence[str]
    results: Sequence[SearchResult]


class TwoStageSearchEngine:
    """
    Handles loading abstract + claim indices and running the two-stage search flow.
    """

    def __init__(self) -> None:
        self._abstract_index: faiss.Index | None = None
        self._abstract_metadata: list[dict] = []
        self._claims_index: faiss.Index | None = None
        self._claims_metadata: list[dict] = []
        self._claim_doc_to_indices: dict[str, list[int]] = {}
        self._doc_metadata: dict[str, dict] = {}
        self._loaded = False
        self._lock = Lock()

    def ensure_loaded(self) -> None:
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            ensure_index_artifacts()

            required = [
                settings.abstract_index_path,
                settings.abstract_metadata_path,
                settings.claims_index_path,
                settings.claims_metadata_path,
                settings.doc_metadata_path,
            ]
            missing = [path for path in required if not path.exists()]
            if missing:
                raise RuntimeError(
                    "FAISS artifacts are missing. Please rebuild the indices. "
                    f"Missing: {', '.join(str(path) for path in missing)}"
                )

            logger.info("Loading abstract index from %s", settings.abstract_index_path)
            self._abstract_index = faiss.read_index(str(settings.abstract_index_path))
            abstract_raw = settings.abstract_metadata_path.read_text(encoding="utf-8")
            self._abstract_metadata = json.loads(abstract_raw)
            if self._abstract_index.ntotal != len(self._abstract_metadata):
                raise RuntimeError(
                    "Abstract metadata length does not match index size "
                    f"({self._abstract_index.ntotal} vs {len(self._abstract_metadata)})."
                )

            logger.info("Loading claim index from %s", settings.claims_index_path)
            self._claims_index = faiss.read_index(str(settings.claims_index_path))
            claims_raw = settings.claims_metadata_path.read_text(encoding="utf-8")
            self._claims_metadata = json.loads(claims_raw)
            if self._claims_index.ntotal != len(self._claims_metadata):
                raise RuntimeError(
                    "Claim metadata length does not match index size "
                    f"({self._claims_index.ntotal} vs {len(self._claims_metadata)})."
                )

            mapping: dict[str, list[int]] = defaultdict(list)
            for idx, meta in enumerate(self._claims_metadata):
                doc_number = meta.get("doc_number", "")
                if doc_number:
                    mapping[doc_number].append(idx)
            self._claim_doc_to_indices = dict(mapping)

            doc_meta_raw = settings.doc_metadata_path.read_text(encoding="utf-8")
            self._doc_metadata = json.loads(doc_meta_raw)

            self._loaded = True

    def search_top_docs(self, query_vector: np.ndarray, top_k: int) -> list[str]:
        self.ensure_loaded()
        if self._abstract_index is None:
            raise RuntimeError("Abstract index not loaded.")

        total = len(self._abstract_metadata)
        if total == 0:
            return []

        k = min(max(top_k, 1), total)
        scores, indices = self._abstract_index.search(query_vector, k)
        doc_numbers: list[str] = []
        seen: set[str] = set()
        for idx in indices[0]:
            if idx < 0 or idx >= total:
                continue
            doc_number = self._abstract_metadata[idx].get("doc_number")
            if not doc_number or doc_number in seen:
                continue
            seen.add(doc_number)
            doc_numbers.append(doc_number)
        return doc_numbers

    def apply_keyword_filter(
        self, doc_numbers: list[str], keyword_filter: str | None
    ) -> list[str]:
        keyword = parse_keyword_filter(keyword_filter)
        if not keyword:
            return doc_numbers

        filtered: list[str] = []
        for doc_number in doc_numbers:
            meta = self._doc_metadata.get(doc_number)
            if not meta:
                continue
            haystack = f"{meta.get('title', '')} {meta.get('abstract', '')}".lower()
            if keyword in haystack:
                filtered.append(doc_number)
        return filtered

    def search_claims_for_docs(
        self,
        doc_numbers: list[str],
        query_vectors: np.ndarray,
        query_texts: Sequence[str],
        top_k: int,
    ) -> List[SearchResult]:
        self.ensure_loaded()
        if self._claims_index is None:
            raise RuntimeError("Claim index not loaded.")

        candidate_indices: list[int] = []
        for doc in doc_numbers:
            candidate_indices.extend(self._claim_doc_to_indices.get(doc, []))

        if not candidate_indices:
            candidate_indices = list(range(len(self._claims_metadata)))

        candidate_vectors = self._reconstruct_vectors(candidate_indices)
        if candidate_vectors.size == 0 or query_vectors.size == 0:
            return []

        # Compute cosine similarity between each query claim and candidate claims.
        similarity_matrix = np.matmul(query_vectors, candidate_vectors.T)

        per_doc_best: dict[str, tuple[float, dict, str | None]] = {}
        for local_idx, original_idx in enumerate(candidate_indices):
            meta = self._claims_metadata[original_idx]
            doc_number = meta.get("doc_number", "N/A")
            if doc_number == "N/A":
                continue

            column_scores = similarity_matrix[:, local_idx]
            best_row = int(np.argmax(column_scores))
            score = float(column_scores[best_row])
            matched_query_text = query_texts[best_row] if 0 <= best_row < len(query_texts) else None

            prev = per_doc_best.get(doc_number)
            if prev is None or score > prev[0]:
                per_doc_best[doc_number] = (score, meta, matched_query_text)

        sorted_docs = sorted(
            per_doc_best.items(),
            key=lambda item: item[1][0],
            reverse=True,
        )

        results: List[SearchResult] = []
        for doc_number, (score, meta, matched_query_text) in sorted_docs[: top_k or len(sorted_docs)]:
            results.append(
                SearchResult(
                    doc_number=doc_number,
                    source_type=meta.get("source_type", "claim"),
                    source_text=meta.get("source_text", ""),
                    score=score,
                    query_claim_text=matched_query_text,
                )
            )
        return results

    def _reconstruct_vectors(self, indices: list[int]) -> np.ndarray:
        assert self._claims_index is not None
        dim = self._claims_index.d
        vectors = np.zeros((len(indices), dim), dtype=np.float32)
        for row, idx in enumerate(indices):
            vectors[row, :] = self._claims_index.reconstruct(int(idx))
        return vectors


_engine: TwoStageSearchEngine | None = None
_engine_lock = Lock()


def get_search_engine() -> TwoStageSearchEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = TwoStageSearchEngine()
    return _engine


def run_semantic_search(
    patent_id: str,
    top_k: int | None = None,
    keyword_filter: str | None = None,
) -> SearchResponse:
    scraped = scrape_patent(patent_id)
    abstract_text = scraped.abstract.strip()
    combined_text = _combine_scraped_text(scraped)
    if not combined_text:
        raise RuntimeError("Scraped patent did not produce any text to embed.")

    if abstract_text:
        abstract_vector = _embed_single_text(abstract_text)
    else:
        logger.warning(
            "Scraped patent %s has no abstract; using combined text for stage-1 filtering.",
            scraped.patent_id,
        )
        abstract_vector = _embed_single_text(combined_text)

    claim_texts = [claim.strip() for claim in scraped.claims if claim.strip()]
    if not claim_texts:
        logger.warning(
            "Scraped patent %s has no claims; using combined text for claim comparison.",
            scraped.patent_id,
        )
        claim_texts = [combined_text]
    claim_vectors = _embed_texts(claim_texts)

    engine = get_search_engine()
    doc_candidates = engine.search_top_docs(
        abstract_vector,
        settings.filter_top_k_docs,
    )
    if not doc_candidates:
        logger.warning(
            "Abstract filtering returned no documents; falling back to entire claim index."
        )

    doc_candidates = engine.apply_keyword_filter(doc_candidates, keyword_filter)
    if keyword_filter and not doc_candidates:
        logger.info(
            "Keyword filter '%s' removed all candidates; returning empty results.",
            keyword_filter,
        )
        return SearchResponse(
            patent_id=scraped.patent_id,
            scraped_title=scraped.title,
            scraped_abstract=scraped.abstract,
            scraped_claims=tuple(claim_texts),
            results=[],
        )

    results = engine.search_claims_for_docs(
        doc_candidates,
        claim_vectors,
        claim_texts,
        top_k or settings.index_top_k,
    )

    return SearchResponse(
        patent_id=scraped.patent_id,
        scraped_title=scraped.title,
        scraped_abstract=scraped.abstract,
        scraped_claims=tuple(claim_texts),
        results=results,
    )


def _embed_single_text(text: str) -> np.ndarray:
    return _embed_texts([text])


def _embed_texts(texts: Sequence[str]) -> np.ndarray:
    vectors = embed_texts(list(texts))
    return normalize_embeddings(vectors)


def _combine_scraped_text(scraped: PatentScrapeResult) -> str:
    parts = [scraped.title, scraped.abstract, " ".join(scraped.claims)]
    combined = ". ".join(part for part in parts if part)
    return combined.strip()


def serialize_search_response(response: SearchResponse) -> dict:
    """Convert dataclass response into JSON-serializable dict."""
    return {
        "patent_id": response.patent_id,
        "scraped_title": response.scraped_title,
        "scraped_abstract": response.scraped_abstract,
        "scraped_claims": list(response.scraped_claims),
        "results": [asdict(result) for result in response.results],
    }


def parse_keyword_filter(filter_value: str | None) -> str | None:
    if not filter_value:
        return None
    cleaned = filter_value.strip()
    if not cleaned:
        return None
    cleaned = cleaned.strip('"').strip("'").strip()
    cleaned = cleaned.lower()
    return cleaned or None
