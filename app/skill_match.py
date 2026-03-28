"""
Skill Matching — Semantic skill comparison
==========================================
Extracts skills from both resume and JD dynamically,
then uses embedding cosine similarity to match them.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
from app.skill_extractor import extract_keywords
from app.embeddings import embed_texts


def dynamic_skill_match(
    resume_text: str,
    jd_text: str,
    match_threshold: float = 0.75,
) -> Tuple[float, List[Dict], List[Dict]]:
    """
    Semantically match skills extracted from resume and JD.

    Returns
    -------
    skill_score : float in [0, 1] — hybrid skill match score
                  (60% match ratio + 40% average match quality)
    matched     : list of {"jd_skill", "resume_skill", "score"}
    missing     : list of {"jd_skill", "best_score"}
    """
    resume_skills = extract_keywords(resume_text, top_n=30)
    jd_skills = extract_keywords(jd_text, top_n=30)

    if not resume_skills or not jd_skills:
        return 0.0, [], [{"jd_skill": s, "best_score": 0.0} for s in jd_skills]

    resume_emb = embed_texts(resume_skills)
    jd_emb = embed_texts(jd_skills)

    sim_matrix = cosine_similarity(resume_emb, jd_emb)

    matched: List[Dict] = []
    missing: List[Dict] = []

    for j_idx, jd_skill in enumerate(jd_skills):
        best_r_idx = int(sim_matrix[:, j_idx].argmax())
        best_score = float(sim_matrix[best_r_idx, j_idx])

        if best_score >= match_threshold:
            matched.append({
                "jd_skill": jd_skill,
                "resume_skill": resume_skills[best_r_idx],
                "score": round(best_score, 3),
            })
        else:
            missing.append({
                "jd_skill": jd_skill,
                "best_score": round(best_score, 3),
            })

    # Hybrid score: breadth (did you match most skills?) + quality (how well?)
    match_ratio = len(matched) / len(jd_skills) if jd_skills else 0.0
    avg_quality = (
        float(np.mean([m["score"] for m in matched])) if matched else 0.0
    )
    skill_score = 0.60 * match_ratio + 0.40 * avg_quality
    return skill_score, matched, missing