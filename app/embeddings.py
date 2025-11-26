from __future__ import annotations

import logging
from threading import Lock
from typing import Iterable, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

_model_lock = Lock()
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info("Loading embedding model: %s", settings.model_name)
                _model = SentenceTransformer(settings.model_name)
    return _model


def embed_texts(
    texts: Sequence[str],
    batch_size: int | None = None,
    show_progress_bar: bool | None = None,
) -> np.ndarray:
    """
    Embed a collection of texts into a numpy array of shape (n, dim).
    """
    if not texts:
        dim = get_model().get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    model = get_model()
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size or settings.embedding_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=show_progress_bar
        if show_progress_bar is not None
        else len(texts) >= 1000,
    )

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    return embeddings


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize embedding rows to enable cosine-similarity with FAISS IP search.
    """
    if vectors.size == 0:
        return vectors

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, np.clip(norms, 1e-12, None), out=vectors)
    return vectors
