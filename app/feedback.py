"""
Feedback Module — LLM-powered reasoning and improvement suggestions
===================================================================
Uses Ollama (local) for:
  1. Per-requirement match explanation
  2. Resume improvement suggestions based on gaps
"""

import re
import ollama
from typing import List, Dict

_MODEL = "qwen2.5:1.5b"


def _extract_score(text: str) -> int | None:
    """
    Robustly extract a numeric score from LLM output.
    Handles formats like:
      SCORE: 75         **Score:** 72       Score = 80
      score: 65/100     Rating: 40          75/100
    """
    # Strip markdown bold/italic markers for easier matching
    clean = text.replace("*", "").replace("_", "")

    # Pattern 1: "SCORE" or "Rating" followed by separator then a number
    m = re.search(
        r'(?:score|rating|match)\s*[:=\-]\s*(\d{1,3})',
        clean, re.IGNORECASE
    )
    if m:
        return int(m.group(1))

    # Pattern 2: number followed by /100 or % anywhere
    m = re.search(r'(\d{1,3})\s*(?:/\s*100|%)', clean)
    if m:
        return int(m.group(1))

    # Pattern 3: First standalone number on a line that starts with score-like word
    for line in clean.split("\n"):
        stripped = line.strip()
        if re.match(r'(?:score|rating|match)', stripped, re.IGNORECASE):
            nums = re.findall(r'\d{1,3}', stripped)
            if nums:
                return int(nums[0])

    return None


def _extract_explanation(text: str) -> str:
    """Extract explanation portion from LLM output."""
    clean = text.replace("*", "").replace("_", "")

    # Try to find text after "EXPLANATION:" or "Explanation:" etc.
    m = re.search(r'explanation\s*[:]\s*(.*)', clean, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # Fallback: everything after the score line
    lines = clean.split("\n")
    capture = False
    parts = []
    for line in lines:
        if capture and line.strip():
            parts.append(line.strip())
        if re.match(r'(?:score|rating)', line.strip(), re.IGNORECASE):
            capture = True
    if parts:
        return " ".join(parts)

    # Last resort: return full text
    return text.strip()


def explain_match(jd_requirement: str, resume_chunk: str) -> Dict[str, str]:
    """
    Ask the LLM whether the candidate's resume chunk satisfies a JD requirement.
    Returns a dict with 'llm_score' (0.0 to 1.0) and 'explanation'.
    """
    prompt = f"""Rate how well this resume section matches the job requirement.

Job Requirement:
{jd_requirement}

Resume Section:
{resume_chunk}

Instructions:
- Give a score from 0 to 100 (0 = no match, 100 = perfect match)
- Be strict: partial matches should get partial scores (e.g. 40-60)
- If the resume section is unrelated, give 0-20
- If it partially matches, give 30-60
- If it matches well but is missing something, give 60-80
- If it fully matches, give 80-100

Reply in this format:
SCORE: [number]
EXPLANATION: [2-3 sentences about what matches and what is missing]"""

    try:
        response = ollama.chat(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 250},
        )
        content = response["message"]["content"].strip()

        # Extract score with robust parser
        score_val = _extract_score(content)

        if score_val is None:
            # Could not parse any score — use a neutral fallback but flag it
            score_val = 50
            explanation = f"(Score parsing failed) {content}"
        else:
            # Clamp between 0 and 100
            score_val = max(0, min(100, score_val))
            explanation = _extract_explanation(content)

        return {"llm_score": round(score_val / 100.0, 2), "explanation": explanation}

    except Exception as e:
        return {"llm_score": 0.50, "explanation": f"LLM unavailable: {e}"}


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