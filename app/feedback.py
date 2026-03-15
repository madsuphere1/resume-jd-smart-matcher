"""
Feedback Module — LLM-powered reasoning and improvement suggestions
===================================================================
Uses Ollama (local) for:
  1. Per-requirement match explanation
  2. Resume improvement suggestions based on gaps
"""

import ollama
from typing import List, Dict

_MODEL = "qwen2.5:1.5b"


def explain_match(jd_requirement: str, resume_chunk: str) -> str:
    """
    Ask the LLM whether the candidate's resume chunk satisfies a JD requirement.
    Returns a concise explanation string.
    """
    prompt = f"""You are an expert ATS resume analyst.

Job Requirement:
{jd_requirement}

Candidate Resume Section:
{resume_chunk}

Task:
In 2-3 sentences, explain whether the candidate satisfies this requirement.
Be specific about what matches and what is missing.
Do NOT use greetings or sign-offs."""

    try:
        response = ollama.chat(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": 150},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"LLM unavailable: {e}"


def generate_improvement_suggestions(
    missing_skills: List[Dict],
    weak_matches: List[Dict],
) -> str:
    """
    Generate actionable resume improvement suggestions based on
    missing skills and weakly matched JD requirements.
    """
    missing_list = ", ".join(m["jd_skill"] for m in missing_skills[:10]) if missing_skills else "None"
    weak_list = "\n".join(
        f"- {w['jd_text']}" for w in weak_matches[:5]
    ) if weak_matches else "None"

    prompt = f"""You are a professional resume coach.

Missing Skills (not found in resume):
{missing_list}

Weakly Matched JD Requirements:
{weak_list}

Task:
Generate exactly 5 concise, actionable bullet-point improvements the candidate
should make to their resume to better match this job description.

Rules:
- Each bullet must start with an action verb
- Be specific, reference the missing skills/requirements above
- Do NOT add greetings, sign-offs, or explanations
- Output ONLY the 5 bullet points

Example format:
- Add a section highlighting experience with ...
- Quantify achievements related to ...
- Include certifications for ...
- Emphasize projects involving ...
- Rewrite summary to mention ..."""

    try:
        response = ollama.chat(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 300},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"LLM unavailable: {e}"


def generate_feedback(score, resume_chunks, jd_chunks, similarity_matrix):
    """Legacy wrapper — kept for backward compatibility with app_ui.py."""
    import numpy as np

    low_match_indices = similarity_matrix.max(axis=0) < 0.65
    missing_requirements = [
        jd_chunks[i]["text"] if isinstance(jd_chunks[i], dict) else jd_chunks[i]
        for i in range(len(jd_chunks))
        if low_match_indices[i]
    ]

    if not missing_requirements:
        return "✅ Strong alignment. Add quantified achievements and metrics to strengthen impact."

    weak = [{"jd_text": r} for r in missing_requirements[:5]]
    return generate_improvement_suggestions([], weak)