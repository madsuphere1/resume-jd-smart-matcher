"""
Resume & JD Chunking — Section-Classified
==========================================
Resume  → list of {"section": str, "text": str}
JD      → list of {"category": str, "text": str}
"""

import re
from typing import List, Dict

# ---------------------------------------------------------------------------
# Section header patterns (case-insensitive)
# ---------------------------------------------------------------------------

_RESUME_SECTION_PATTERNS: List[tuple] = [
    (r"\b(professional\s+)?summary\b|\bobjective\b|\bprofile\b", "SUMMARY"),
    (r"\b(work\s+)?experience\b|\bemployment\b|\bwork\s*history\b", "EXPERIENCE"),
    (r"\bprojects?\b|\bportfolio\b", "PROJECTS"),
    (r"\b(technical\s+)?skills?\b|\bcompetenc(ies|e)\b|\btechnolog(ies|y)\b|\btools?\b", "SKILLS"),
    (r"\beducation\b|\bacademic\b|\bqualification\b", "EDUCATION"),
    (r"\bcertification\b|\blicen[sc]e\b|\baccreditation\b|\bcourses?\b|\btraining\b", "CERTIFICATIONS"),
    (r"\bachievement\b|\baward\b|\bhonor\b|\brecognition\b", "ACHIEVEMENTS"),
    (r"\bpublication\b|\bresearch\b", "PUBLICATIONS"),
    (r"\bvolunteer\b|\bcommunity\b|\bextracurricular\b", "VOLUNTEER"),
    (r"\blanguage\b|\binterest\b|\bhobbies?\b", "OTHER"),
]

_JD_NOISE_PATTERNS = re.compile(
    r"^\s*("
    # ── General job portal metadata ──
    r"role\s*category|industry\s*(type)?|department|employment\s*type|"
    r"posted\s*(on|date)?|application\s*deadline|location|salary|"
    r"about\s*(the\s+)?company|company\s+overview|who\s+we\s+are|"
    r"equal\s+opportunity|benefits|perks|what\s+we\s+offer|"
    r"job\s*code|job\s*id|requisition|reference\s*(number|id)|"
    r"number\s+of\s+openings?|vacancies|positions?\s+available|"
    r"work\s*mode|work\s+from\s+home|remote|hybrid|on[\s\-]?site|"
    r"shift\s*timing|notice\s+period|"
    # ── Indian job portal fields (Naukri, etc.) ──
    r"role|functional\s*area|key\s*skills?|"
    r"pg\s*:|ug\s*:|doctorate|education\s*:|"
    r"any\s+graduate|any\s+postgraduate|"
    r"b\.?e\.?|b\.?tech|m\.?tech|m\.?s\.?|m\.?b\.?a\.?|b\.?sc|m\.?sc|"
    r"specialization|"
    r"company\s+profile|about\s+recruiter|recruiter|"
    r"walk[\s\-]?in|interview\s+(date|venue|location)|"
    r"contact\s+(person|email|number)|"
    r"apply\s*(now|here|online|before)|"
    # ── Other noise ──
    r"follow\s+us|connect\s+with\s+us|visit\s+us|"
    r"disclaimer|copyright|all\s+rights\s+reserved|"
    r"we\s+are\s+an?\s+(equal|diverse)|"
    r"accommodation|disability|"
    r"this\s+(job|position)\s+(is|was)\s+posted"
    r")\s*[:|\-]?\s*",
    re.IGNORECASE,
)

# Short key: value lines that are metadata, not requirements
_JD_META_KV_PATTERN = re.compile(
    r"^\s*[A-Za-z\s]{2,25}\s*:\s*.{1,50}\s*$"
)

# Common meta key prefixes
_JD_META_KEYS = re.compile(
    r"^\s*("
    r"role|designation|title|position|grade|level|band|"
    r"pg|ug|education|degree|qualification|"
    r"experience|exp|years|"
    r"location|city|state|country|region|"
    r"salary|ctc|compensation|stipend|package|"
    r"industry|domain|sector|vertical|"
    r"department|function|functional\s*area|"
    r"employment\s*type|job\s*type|engagement|"
    r"shift|notice\s*period|joining|"
    r"vacancies?|openings?|headcount|"
    r"posted|updated|last\s*date|deadline|"
    r"company|organization|employer|recruiter|"
    r"ref|job\s*id|job\s*code|requisition|"
    r"work\s*mode|work\s*type|"
    r"age|gender|nationality|visa"
    r")\s*[:|\-]\s*",
    re.IGNORECASE,
)

_JD_CATEGORY_PATTERNS: List[tuple] = [
    (r"\bresponsibilit(ies|y)\b|\bwhat\s+you.ll\s+do\b|\bday[\s\-]to[\s\-]day\b|\bkey\s+duties\b", "RESPONSIBILITIES"),
    (r"\brequirement\b|\bwhat\s+we.*(look|need)\b|\bmust\s+have\b|\bminimum\b", "REQUIREMENTS"),
    (r"\bskills?\s+(required|needed|expected)\b|\btechnical\s+proficiency\b", "SKILLS"),
    (r"\bqualification\b|\beducation(al)?\s+requirement\b|\bdegree\s+required\b", "QUALIFICATIONS"),
    (r"\btools?\s+(and\s+)?technolog\b|\btech\s+stack\b|\bplatform\b|\bsoftware\s+used\b", "TOOLS"),
    (r"\bnice[\s\-]to[\s\-]have\b|\bpreferred\b|\bbonus\b|\bgood\s+to\s+have\b", "PREFERRED"),
    (r"\bjob\s+description\b|\babout\s+(the\s+)?role\b|\boverview\b", "REQUIREMENTS"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_resume_section(line: str) -> str | None:
    """Return section label if *line* looks like a resume section header."""
    clean = line.strip()
    if len(clean) > 80:  # headers are short
        return None
    for pattern, label in _RESUME_SECTION_PATTERNS:
        if re.search(pattern, clean, re.IGNORECASE):
            return label
    return None


def _detect_jd_category(line: str) -> str | None:
    """Return JD category if *line* looks like a JD section header."""
    clean = line.strip()
    for pattern, label in _JD_CATEGORY_PATTERNS:
        if re.search(pattern, clean, re.IGNORECASE):
            return label
    return None


def _is_jd_noise(line: str) -> bool:
    """True if the line is a meta-field we should ignore."""
    clean = line.strip()
    if not clean:
        return True

    # Direct noise pattern match
    if _JD_NOISE_PATTERNS.match(clean):
        return True

    # Short key: value metadata lines (e.g. "PG: Any Postgraduate")
    if _JD_META_KEYS.match(clean):
        return True

    # Very short lines that are likely labels (< 5 words, ends with colon)
    words = clean.split()
    if len(words) <= 4 and clean.endswith(":"):
        return True

    return False


def _is_low_quality_chunk(text: str) -> bool:
    """
    Reject chunks that are just metadata labels joined together,
    not actual job requirements.
    """
    words = text.split()

    # Too short to be meaningful
    if len(words) < 5:
        return True

    # Count how many words are actual English content vs. labels
    # If most of the chunk is short key:value pairs, reject it
    lines = text.split(".")
    meta_lines = sum(1 for l in lines if _JD_META_KEYS.match(l.strip()) or len(l.strip().split()) < 3)
    if len(lines) > 1 and meta_lines / len(lines) > 0.6:
        return True

    return False


def _approx_tokens(text: str) -> int:
    return len(text.split())


def _sub_chunk(text: str, section: str, max_tokens: int = 300) -> List[Dict]:
    """Split a single section blob into ≤ max_tokens sub-chunks."""
    words = text.split()
    if len(words) <= max_tokens:
        return [{"section": section, "text": text.strip()}]

    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk_text = " ".join(words[i : i + max_tokens]).strip()
        if chunk_text:
            chunks.append({"section": section, "text": chunk_text})
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def smart_chunk_resume(text: str) -> List[Dict]:
    """
    Chunk resume text into labelled sections.

    Returns [{"section": "SKILLS", "text": "..."}, ...]
    """
    lines = text.split("\n")
    sections: List[Dict] = []
    current_section = "SUMMARY"  # default if no header found at start
    current_lines: List[str] = []

    for line in lines:
        detected = _detect_resume_section(line)
        if detected:
            # flush previous section
            blob = " ".join(current_lines).strip()
            if blob and len(blob) > 15:
                sections.extend(_sub_chunk(blob, current_section))
            current_section = detected
            current_lines = []
        else:
            clean = line.strip()
            if clean:
                current_lines.append(clean)

    # flush last section
    blob = " ".join(current_lines).strip()
    if blob and len(blob) > 15:
        sections.extend(_sub_chunk(blob, current_section))

    return sections


def smart_chunk_jd(text: str) -> List[Dict]:
    """
    Chunk JD text into labelled requirement categories.

    Returns [{"category": "REQUIREMENTS", "text": "..."}, ...]
    Filters out noise lines (role category, industry, etc.).
    """
    lines = text.split("\n")
    chunks: List[Dict] = []
    current_category = "REQUIREMENTS"  # default
    current_lines: List[str] = []

    for line in lines:
        if _is_jd_noise(line):
            continue

        detected = _detect_jd_category(line)
        if detected:
            # flush
            blob = " ".join(current_lines).strip()
            if blob and len(blob) > 30 and not _is_low_quality_chunk(blob):
                for sub in _sub_chunk(blob, current_category, max_tokens=300):
                    chunks.append({"category": sub["section"], "text": sub["text"]})
            current_category = detected
            current_lines = []
        else:
            clean = line.strip()
            # Keep only meaningful lines (> 20 chars to skip short labels)
            if clean and len(clean) > 20:
                current_lines.append(clean)

    # flush last
    blob = " ".join(current_lines).strip()
    if blob and len(blob) > 30 and not _is_low_quality_chunk(blob):
        for sub in _sub_chunk(blob, current_category, max_tokens=300):
            chunks.append({"category": sub["section"], "text": sub["text"]})

    return chunks