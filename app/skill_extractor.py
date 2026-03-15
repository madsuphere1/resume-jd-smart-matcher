"""
Skill Extractor — Dynamic keyword extraction via KeyBERT
========================================================
Shares the SentenceTransformer model from embeddings module.
"""

from keybert import KeyBERT
from typing import List
from app.embeddings import get_model

_kw_model = None


def _get_kw_model() -> KeyBERT:
    """Lazy-load KeyBERT using the shared embedding model."""
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT(model=get_model())
    return _kw_model


def extract_keywords(text: str, top_n: int = 25) -> List[str]:
    """
    Dynamically extract skill-like keyphrases from text.

    Uses MMR (Maximal Marginal Relevance) diversity to avoid near-duplicate phrases.

    Returns list of lowercase keyphrases.
    """
    model = _get_kw_model()
    keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=top_n,
        use_mmr=True,
        diversity=0.5,
    )
    return [kw[0].lower() for kw in keywords]