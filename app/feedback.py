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


def explain_match(jd_requirement: str, resume_chunk: str) -> Dict[str, str]:
    """
    Ask the LLM whether the candidate's resume chunk satisfies a JD requirement.
    Returns a dict with 'llm_score' (0.0 to 1.0) and 'explanation'.
    """
    prompt = f"""You are an expert strict ATS resume analyst.

Job Requirement:
{jd_requirement}

Candidate Resume Section:
{resume_chunk}

Task:
Evaluate how well the candidate satisfies this EXACT requirement.
Be strict: if they mention "Python" but the requirement is "Python and AWS", and they don't have AWS, the score should drop significantly based on importance.

Format your response EXACTLY like this:
SCORE: [An integer from 0 to 100]
EXPLANATION: [2-3 sentences explaining the score, specifically noting missing or irrelevant details.]"""

    default_res = {"llm_score": 0.50, "explanation": "LLM extraction failed."}

    try:
        response = ollama.chat(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 150},
        )
        content = response["message"]["content"].strip()
        
        # Parse output
        lines = content.split("\n")
        score_val = 50
        explanation = content
        
        for line in lines:
            if line.upper().startswith("SCORE:"):
                s_str = line.split(":", 1)[1].strip()
                import re
                nums = re.findall(r'\d+', s_str)
                if nums:
                    score_val = int(nums[0])
            elif line.upper().startswith("EXPLANATION:"):
                explanation = line.split(":", 1)[1].strip()
                
        # If explanation is multi-line grab the rest
        if "EXPLANATION:" in content:
            explanation = content.split("EXPLANATION:", 1)[1].strip()

        # Clamp between 0 and 100, then convert to 0.0 - 1.0
        score_val = max(0, min(100, score_val))
        return {"llm_score": round(score_val / 100.0, 2), "explanation": explanation}
    except Exception as e:
        default_res["explanation"] = f"LLM unavailable: {e}"
        return default_res


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