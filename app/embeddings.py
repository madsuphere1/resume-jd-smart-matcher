"""
Embedding Module — Lazy-loaded SentenceTransformer
===================================================
Uses BAAI/bge-small-en for all embeddings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union

_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-small-en")
    return _model


def embed_texts(texts: Union[List[str], List[Dict]]) -> np.ndarray:
    """
    Embed a list of strings OR a list of chunk dicts (uses "text" key).

    Returns numpy array of shape (n, dim) with L2-normalized embeddings.
    """
    if not texts:
        return np.array([])

    # Accept both plain strings and chunk dicts
    if isinstance(texts[0], dict):
        raw = [t["text"] for t in texts]
    else:
        raw = list(texts)

    model = get_model()
    return model.encode(raw, normalize_embeddings=True)