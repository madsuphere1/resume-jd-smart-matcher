"""
ChromaDB Vector Store — In-Memory Resume Chunk Storage
======================================================
Stores resume chunks with section metadata for RAG retrieval.
"""

import chromadb
import numpy as np
from typing import List, Dict, Optional


class ResumeVectorStore:
    """
    In-memory ChromaDB-backed vector store for resume chunks.

    Each chunk is stored with:
      - id:        unique chunk identifier
      - embedding: dense vector from SentenceTransformer
      - document:  the chunk text
      - metadata:  {"section": "SKILLS" | "EXPERIENCE" | ...}
    """

    def __init__(self, collection_name: str = "resume_chunks"):
        self._client = chromadb.Client()               # in-memory, no disk state
        self._collection_name = collection_name
        self._collection = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self):
        """Create or reset the collection."""
        # Delete existing to avoid stale data from previous runs
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_chunks(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
    ) -> None:
        """
        Store resume chunks in ChromaDB.

        Parameters
        ----------
        chunks : list of {"section": str, "text": str}
        embeddings : np.ndarray of shape (n, dim)
        """
        self._ensure_collection()

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        metadatas = [{"section": c["section"]} for c in chunks]
        embedding_list = embeddings.tolist()

        self._collection.add(
            ids=ids,
            embeddings=embedding_list,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        section_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve the top-k most similar resume chunks.

        Parameters
        ----------
        query_embedding : 1-D numpy array (dim,)
        top_k : number of results
        section_filter : optional section label to restrict results

        Returns
        -------
        List of {"text": str, "section": str, "score": float}
        """
        if self._collection is None:
            return []

        where_filter = {"section": section_filter} if section_filter else None

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count()),
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB cosine distance = 1 - cosine_similarity
                hits.append({
                    "text": doc,
                    "section": meta["section"],
                    "score": round(1.0 - dist, 4),
                })
        return hits

    def get_all_chunks(self) -> List[Dict]:
        """Return all stored chunks with their metadata."""
        if self._collection is None or self._collection.count() == 0:
            return []
        all_data = self._collection.get(include=["documents", "metadatas"])
        return [
            {"text": doc, "section": meta["section"]}
            for doc, meta in zip(all_data["documents"], all_data["metadatas"])
        ]
