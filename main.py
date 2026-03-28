"""
Resume ↔ JD Matching Pipeline
==============================
End-to-end orchestration: extract → chunk → embed → store → match → score → reason.
"""

from app.chunking import smart_chunk_resume, smart_chunk_jd
from app.embeddings import embed_texts
from app.vector_store import ResumeVectorStore
from app.similarity import compute_section_similarity
from app.skill_match import dynamic_skill_match
from app.feedback import explain_match, generate_improvement_suggestions
from typing import Dict, List
import numpy as np


def run(resume_text: str, jd_text: str, enable_llm: bool = True) -> Dict:
    """
    Full matching pipeline.

    Returns a structured result dict with:
      - final_score, semantic_score, skill_score, requirement_coverage
      - section_grid  (resume_section × jd_category → cosine score)
      - per_jd_matches  (per requirement best match)
      - matched_skills, missing_skills
      - rag_results  (ChromaDB retrieval per JD requirement)
      - explanations  (LLM reasoning per top match)
      - improvement_suggestions  (LLM-generated)
      - resume_chunks, jd_chunks
    """

    # -------- Phase 1: Chunking --------
    resume_chunks = smart_chunk_resume(resume_text)
    jd_chunks = smart_chunk_jd(jd_text)

    if not resume_chunks:
        return {"error": "Could not extract meaningful content from the resume."}
    if not jd_chunks:
        return {"error": "Could not extract requirements from the job description."}

    # -------- Phase 2: Embeddings --------
    resume_embeddings = embed_texts(resume_chunks)
    jd_embeddings = embed_texts(jd_chunks)

    # -------- Phase 3: Vector Store (ChromaDB) --------
    vector_store = ResumeVectorStore()
    vector_store.store_chunks(resume_chunks, resume_embeddings)

    # -------- Phase 4: RAG Retrieval --------
    rag_results: List[Dict] = []
    for j_idx, jd_chunk in enumerate(jd_chunks):
        jd_emb = jd_embeddings[j_idx]
        hits = vector_store.query(jd_emb, top_k=3)
        rag_results.append({
            "jd_text": jd_chunk["text"],
            "jd_category": jd_chunk.get("category", "REQUIREMENTS"),
            "retrieved_chunks": hits,
        })

    # -------- Phase 5: Section-wise Similarity --------
    sim_result = compute_section_similarity(
        resume_chunks, jd_chunks,
        resume_embeddings, jd_embeddings,
    )

    semantic_score = sim_result["semantic_score"]
    requirement_coverage = sim_result["requirement_coverage"]

    # -------- Phase 6: Skill Matching --------
    skill_score, matched_skills, missing_skills = dynamic_skill_match(
        resume_text, jd_text
    )

    # -------- Phase 7: LLM Reasoning (optional) --------
    explanations: List[Dict] = []
    improvement_suggestions = ""

    if enable_llm:
        # Explain top 5 matches
        top_matches = sorted(
            sim_result["per_jd_matches"],
            key=lambda x: x["similarity_score"],
            reverse=True,
        )[:5]

        for match in top_matches:
            llm_result = explain_match(
                match["jd_text"],
                match["matched_resume_text"],
            )
            explanations.append({
                "jd_text": match["jd_text"],
                "resume_text": match["matched_resume_text"],
                "score": match["similarity_score"],
                "llm_score": llm_result["llm_score"],
                "explanation": llm_result["explanation"],
            })

        # Improvement suggestions
        weak_matches = [
            m for m in sim_result["per_jd_matches"]
            if m["similarity_score"] < 0.50
        ]
        improvement_suggestions = generate_improvement_suggestions(
            missing_skills, weak_matches
        )

    # -------- Phase 8: Score Composition --------
    # Now that skill_score and requirement_coverage are quality-weighted
    # (not trivially 1.0), we use a direct weighted combination.
    base_score = (
        0.50 * semantic_score
        + 0.30 * skill_score
        + 0.20 * requirement_coverage
    )

    # 2. Integrate LLM Intelligence
    # If LLM reasoning is active, it generated scores (0.0 to 1.0) for the top reqs 
    # based on actual logic (missing/irrelevant context). Blend it with the base score.
    final_score = base_score
    avg_llm_score = 0.0
    
    if enable_llm and explanations:
        avg_llm_score = sum(exp["llm_score"] for exp in explanations) / len(explanations)
        # Blend: 60% traditional vector engine, 40% LLM intelligence
        final_score = (base_score * 0.60) + (avg_llm_score * 0.40)

    # -------- Build Result --------
    return {
        "final_score": round(final_score, 4),
        "semantic_score": round(semantic_score, 4),
        "skill_score": round(skill_score, 4),
        "requirement_coverage": round(requirement_coverage, 4),
        "llm_alignment_score": round(avg_llm_score, 4) if explanations else None,
        "section_grid": sim_result["section_grid"],
        "resume_sections": sim_result["resume_sections"],
        "jd_categories": sim_result["jd_categories"],
        "per_jd_matches": sim_result["per_jd_matches"],
        "rag_results": rag_results,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "explanations": explanations,
        "improvement_suggestions": improvement_suggestions,
        "resume_chunks": resume_chunks,
        "jd_chunks": jd_chunks,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open("data/sample_resume.txt") as f:
        resume_text = f.read()
    with open("data/sample_jd.txt") as f:
        jd_text = f.read()

    result = run(resume_text, jd_text, enable_llm=False)

    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        print(f"\n{'='*60}")
        print(f"  MATCH SCORE: {result['final_score']*100:.1f}%")
        print(f"{'='*60}")
        print(f"  Semantic:   {result['semantic_score']*100:.1f}%")
        print(f"  Skill:      {result['skill_score']*100:.1f}%")
        print(f"  Coverage:   {result['requirement_coverage']*100:.1f}%")
        print(f"{'='*60}")

        print("\n--- Section Mapping Grid ---")
        for (r_sec, j_cat), score in sorted(result["section_grid"].items()):
            print(f"  {r_sec:20s} <-> {j_cat:20s}  ->  {score:.3f}")

        print(f"\n--- Matched Skills ({len(result['matched_skills'])}) ---")
        for s in result["matched_skills"]:
            print(f"  [OK] {s['jd_skill']} <- {s['resume_skill']} ({s['score']:.2f})")

        print(f"\n--- Missing Skills ({len(result['missing_skills'])}) ---")
        for s in result["missing_skills"]:
            print(f"  [X]  {s['jd_skill']} (best match: {s['best_score']:.2f})")