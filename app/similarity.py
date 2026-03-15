"""
Similarity Module — Section-wise Full Outer Join
=================================================
Computes cosine similarity between resume sections and JD categories,
producing a section mapping grid (full outer join style).
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
from collections import defaultdict


def compute_section_similarity(
    resume_chunks: List[Dict],
    jd_chunks: List[Dict],
    resume_embeddings: np.ndarray,
    jd_embeddings: np.ndarray,
    match_threshold: float = 0.50,
) -> Dict:
    """
    Section-wise full outer join similarity.

    For every (resume_section, jd_category) pair, compute the best cosine
    similarity among constituent chunks.  Also compute aggregate scores.

    Returns
    -------
    dict with keys:
        section_grid    : dict[(resume_section, jd_category)] → best cosine score
        semantic_score  : float — mean of per-JD-requirement best-match scores
        requirement_coverage : float — fraction of JD reqs matched above threshold
        per_jd_matches  : list of per-JD-requirement best match dicts
        similarity_matrix : np.ndarray (n_resume × n_jd)
        resume_sections : sorted list of unique resume section labels
        jd_categories   : sorted list of unique JD category labels
    """
    if resume_embeddings.size == 0 or jd_embeddings.size == 0:
        return {
            "section_grid": {},
            "semantic_score": 0.0,
            "requirement_coverage": 0.0,
            "per_jd_matches": [],
            "similarity_matrix": np.array([]),
            "resume_sections": [],
            "jd_categories": [],
        }

    # Full cosine similarity matrix  (n_resume × n_jd)
    sim_matrix = cosine_similarity(resume_embeddings, jd_embeddings)

    # ---- Section mapping grid (full outer join) ----
    resume_section_labels = [c["section"] for c in resume_chunks]
    jd_category_labels = [c.get("category", "REQUIREMENTS") for c in jd_chunks]

    unique_resume_sections = sorted(set(resume_section_labels))
    unique_jd_categories = sorted(set(jd_category_labels))

    # Group chunk indices by their section / category
    resume_idx_by_section: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(resume_section_labels):
        resume_idx_by_section[s].append(i)

    jd_idx_by_category: Dict[str, List[int]] = defaultdict(list)
    for j, c in enumerate(jd_category_labels):
        jd_idx_by_category[c].append(j)

    section_grid: Dict[Tuple[str, str], float] = {}
    for r_sec in unique_resume_sections:
        for j_cat in unique_jd_categories:
            r_indices = resume_idx_by_section[r_sec]
            j_indices = jd_idx_by_category[j_cat]
            # Sub-matrix for this (section, category) pair
            sub = sim_matrix[np.ix_(r_indices, j_indices)]
            section_grid[(r_sec, j_cat)] = float(sub.max()) if sub.size else 0.0

    # ---- Per-JD-requirement best match ----
    per_jd_matches = []
    for j_idx in range(len(jd_chunks)):
        best_r_idx = int(sim_matrix[:, j_idx].argmax())
        best_score = float(sim_matrix[best_r_idx, j_idx])
        per_jd_matches.append({
            "jd_text": jd_chunks[j_idx]["text"],
            "jd_category": jd_chunks[j_idx].get("category", "REQUIREMENTS"),
            "matched_resume_text": resume_chunks[best_r_idx]["text"],
            "matched_resume_section": resume_chunks[best_r_idx]["section"],
            "similarity_score": round(best_score, 4),
        })

    # ---- Aggregate scores ----
    best_per_jd = sim_matrix.max(axis=0)  # best resume match for each JD chunk
    semantic_score = float(np.mean(best_per_jd))
    requirement_coverage = float(np.mean(best_per_jd >= match_threshold))

    return {
        "section_grid": section_grid,
        "semantic_score": semantic_score,
        "requirement_coverage": requirement_coverage,
        "per_jd_matches": per_jd_matches,
        "similarity_matrix": sim_matrix,
        "resume_sections": unique_resume_sections,
        "jd_categories": unique_jd_categories,
    }


# ---------------------------------------------------------------------------
# Legacy compatibility wrappers
# ---------------------------------------------------------------------------

def compute_similarity(resume_embeddings, jd_embeddings):
    """Original API — returns (score, matrix)."""
    matrix = cosine_similarity(resume_embeddings, jd_embeddings)
    max_scores = matrix.max(axis=0)
    return float(np.mean(max_scores)), matrix


def get_top_matches(resume_chunks, jd_chunks, similarity_matrix) -> List[Dict]:
    """Original API — returns list sorted by score desc."""
    results = []
    for j_idx in range(len(jd_chunks)):
        best_r = int(similarity_matrix[:, j_idx].argmax())
        best_s = float(similarity_matrix[best_r, j_idx])
        r_text = resume_chunks[best_r]["text"] if isinstance(resume_chunks[best_r], dict) else resume_chunks[best_r]
        j_text = jd_chunks[j_idx]["text"] if isinstance(jd_chunks[j_idx], dict) else jd_chunks[j_idx]
        results.append({
            "jd_requirement": j_text,
            "matched_resume_section": r_text,
            "similarity_score": best_s,
        })
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results