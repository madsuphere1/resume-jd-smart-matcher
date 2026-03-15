"""
Resume ↔ JD Matcher — Streamlit Dashboard
==========================================
Premium UI with section mapping grid, skill analysis, and LLM reasoning.
"""

import streamlit as st
import requests
import trafilatura
from bs4 import BeautifulSoup
import pdfplumber
from docx import Document
import pytesseract
import os
import numpy as np
import cv2
from pdf2image import convert_from_bytes
from main import run as run_pipeline
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pytesseract.pytesseract.tesseract_cmd = os.path.join(
    os.getcwd(), "tesseract", "tesseract.exe"
)

# ═══════════════════════════════════════════════════════════════════════════
# FILE PARSING
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        pass

    if len(text.strip()) < 100:
        file.seek(0)
        poppler_path = os.path.join(
            BASE_DIR, "tesseract", "poppler", "Library", "bin"
        )
        try:
            images = convert_from_bytes(file.read(), poppler_path=poppler_path)
            for img in images:
                img_arr = np.array(img)
                gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
                text += pytesseract.image_to_string(gray) + "\n"
        except Exception:
            pass

    return text


def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


def fetch_jd_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        downloaded = trafilatura.extract(response.text)
        if downloaded and len(downloaded) > 200:
            return downloaded
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.extract()
        text = soup.get_text(separator="\n")
        return text if len(text) > 200 else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Score gauge */
    .score-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin-bottom: 1.5rem;
    }
    .score-value {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e94560, #ff6b6b, #ffd93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .score-label {
        font-size: 1rem;
        opacity: 0.8;
        margin-top: 0.3rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .sub-score {
        display: inline-block;
        background: rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        margin: 0.3rem;
        font-size: 0.85rem;
    }
    .sub-score strong {
        color: #ffd93d;
    }

    /* Skill chips */
    .skill-matched {
        display: inline-block;
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .skill-missing {
        display: inline-block;
        background: linear-gradient(135deg, #e94560, #c62828);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Section header */
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    }

    /* Match card */
    .match-card {
        background: #f8f9ff;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        color: #1a1a2e;
    }
    .match-card strong, .match-card em, .match-card small {
        color: #1a1a2e;
    }
    .match-card-weak {
        background: #fff5f5;
        border-left: 4px solid #e94560;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        color: #1a1a2e;
    }
    .match-card-weak strong, .match-card-weak em, .match-card-weak small {
        color: #1a1a2e;
    }

    /* RAG card */
    .rag-card {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
        color: #1a1a2e;
    }
    .rag-card strong {
        color: #166534;
    }

    div[data-testid="stDataFrame"] > div {
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Resume ↔ JD Matcher",
    page_icon="🎯",
    layout="wide",
)

inject_css()

st.markdown("# 🎯 Resume ↔ JD Smart Matcher")
st.markdown(
    "*Fully local • RAG-powered • Section-wise semantic matching • LLM reasoning*"
)
st.markdown("---")

# ─── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    enable_llm = st.toggle("Enable LLM Reasoning", value=True)
    st.caption("Uses Ollama (qwen2.5:1.5b) for explanations")
    st.markdown("---")
    st.markdown("### 📊 Scoring Weights")
    w_semantic = st.slider("Semantic Similarity", 0.0, 1.0, 0.50, 0.05)
    w_skill = st.slider("Skill Overlap", 0.0, 1.0, 0.30, 0.05)
    w_coverage = st.slider("Requirement Coverage", 0.0, 1.0, 0.20, 0.05)
    total_w = w_semantic + w_skill + w_coverage
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"Weights sum to {total_w:.2f} — should be 1.0")

# ─── Input Section ────────────────────────────────────────────────────────
col_resume, col_jd = st.columns(2)

with col_resume:
    st.markdown("### 📄 Resume")
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF / DOCX / TXT)",
        type=["pdf", "docx", "txt"],
    )

with col_jd:
    st.markdown("### 💼 Job Description")
    jd_option = st.radio(
        "Input method:",
        ["Paste Text", "Public URL"],
        horizontal=True,
    )
    jd_text = ""
    if jd_option == "Paste Text":
        jd_text = st.text_area("Paste JD here", height=200)
    else:
        jd_url = st.text_input("Enter public JD URL")
        if jd_url:
            fetched = fetch_jd_from_url(jd_url)
            if fetched:
                st.success("✅ JD fetched successfully!")
                jd_text = fetched
            else:
                st.warning(
                    "⚠ Could not scrape this URL (may block bots). Paste manually:"
                )
                jd_text = st.text_area("Paste JD here", height=200)

st.markdown("---")

# ─── Analyze Button ──────────────────────────────────────────────────────
if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):

    if not uploaded_file:
        st.warning("Please upload a resume.")
        st.stop()
    if not jd_text or not jd_text.strip():
        st.warning("Please provide a Job Description.")
        st.stop()

    # ── Extract resume text ──
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
        if len(resume_text.strip()) < 80:
            st.error(
                "⚠ Scanned/image-based PDF with too little text. "
                "Please upload a DOCX or text-based PDF."
            )
            st.stop()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        resume_text = uploaded_file.read().decode("utf-8")

    # ── Run pipeline ──
    with st.spinner("🔄 Analyzing... (embedding → storing → matching → reasoning)"):
        result = run_pipeline(resume_text, jd_text, enable_llm=enable_llm)

    if "error" in result:
        st.error(result["error"])
        st.stop()

    # Recompute final score with user-configured weights
    final_score = (
        w_semantic * result["semantic_score"]
        + w_skill * result["skill_score"]
        + w_coverage * result["requirement_coverage"]
    )

    # ══════════════════════════════════════════════════════════════════
    # RESULTS DASHBOARD
    # ══════════════════════════════════════════════════════════════════

    # ── Score Card ──
    st.markdown(f"""
    <div class="score-card">
        <div class="score-value">{final_score*100:.0f}%</div>
        <div class="score-label">Overall Match Score</div>
        <div style="margin-top: 1rem;">
            <span class="sub-score">🧠 Semantic: <strong>{result['semantic_score']*100:.0f}%</strong></span>
            <span class="sub-score">🔧 Skill: <strong>{result['skill_score']*100:.0f}%</strong></span>
            <span class="sub-score">📋 Coverage: <strong>{result['requirement_coverage']*100:.0f}%</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Chunk counts ──
    c1, c2 = st.columns(2)
    c1.metric("Resume Chunks", len(result["resume_chunks"]))
    c2.metric("JD Requirements", len(result["jd_chunks"]))

    # ══════════════════════════════════════════════════════════════════
    # SECTION MAPPING GRID (Full Outer Join Heatmap)
    # ══════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">📐 Section Mapping Grid (Resume ↔ JD)</div>', unsafe_allow_html=True)
    st.caption("Each cell shows the best cosine similarity between that resume section and JD category.")

    grid = result["section_grid"]
    r_secs = result["resume_sections"]
    j_cats = result["jd_categories"]

    if grid and r_secs and j_cats:
        df_grid = pd.DataFrame(
            [[grid.get((r, j), 0.0) for j in j_cats] for r in r_secs],
            index=r_secs,
            columns=j_cats,
        )

        def color_cells(val):
            if val >= 0.70:
                return "background-color: #22c55e; color: white; font-weight: 600"
            elif val >= 0.50:
                return "background-color: #fbbf24; color: #1a1a2e; font-weight: 500"
            elif val >= 0.30:
                return "background-color: #fb923c; color: white; font-weight: 500"
            else:
                return "background-color: #ef4444; color: white; font-weight: 500"

        styled = df_grid.style.format("{:.3f}").map(color_cells)
        st.dataframe(styled, use_container_width=True, height=min(400, 60 * len(r_secs) + 60))

        # Feasibility per section
        st.markdown("**Section-level Feasibility:**")
        for r_sec in r_secs:
            row_scores = [grid.get((r_sec, j), 0.0) for j in j_cats]
            avg = np.mean(row_scores) if row_scores else 0.0
            icon = "✅" if avg >= 0.50 else ("⚠️" if avg >= 0.30 else "❌")
            st.markdown(f"{icon} **{r_sec}** — avg match: `{avg:.3f}`")

    # ══════════════════════════════════════════════════════════════════
    # SKILL ANALYSIS
    # ══════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">🔧 Skill Analysis</div>', unsafe_allow_html=True)

    col_m, col_x = st.columns(2)

    with col_m:
        st.markdown(f"**✅ Matched Skills** ({len(result['matched_skills'])})")
        if result["matched_skills"]:
            chips = ""
            for s in result["matched_skills"]:
                chips += f'<span class="skill-matched">{s["jd_skill"]} ← {s["resume_skill"]} ({s["score"]:.0%})</span> '
            st.markdown(chips, unsafe_allow_html=True)

            # Table view
            with st.expander("View details"):
                df_matched = pd.DataFrame(result["matched_skills"])
                df_matched.columns = ["JD Skill", "Resume Skill", "Similarity"]
                st.dataframe(df_matched, use_container_width=True, hide_index=True)
        else:
            st.info("No skills matched above threshold.")

    with col_x:
        st.markdown(f"**❌ Missing Skills** ({len(result['missing_skills'])})")
        if result["missing_skills"]:
            chips = ""
            for s in result["missing_skills"]:
                chips += f'<span class="skill-missing">{s["jd_skill"]} ({s["best_score"]:.0%})</span> '
            st.markdown(chips, unsafe_allow_html=True)

            with st.expander("View details"):
                df_missing = pd.DataFrame(result["missing_skills"])
                df_missing.columns = ["JD Skill", "Best Match Score"]
                st.dataframe(df_missing, use_container_width=True, hide_index=True)
        else:
            st.success("All JD skills matched!")

    # ══════════════════════════════════════════════════════════════════
    # PER-REQUIREMENT MATCHES
    # ══════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">🔎 Per-Requirement Match Details</div>', unsafe_allow_html=True)

    sorted_matches = sorted(
        result["per_jd_matches"],
        key=lambda x: x["similarity_score"],
        reverse=True,
    )

    for i, match in enumerate(sorted_matches):
        score = match["similarity_score"]
        card_class = "match-card" if score >= 0.50 else "match-card-weak"
        icon = "✅" if score >= 0.70 else ("⚠️" if score >= 0.50 else "❌")

        st.markdown(f"""
        <div class="{card_class}">
            <strong>{icon} [{match['jd_category']}] {match['jd_text'][:120]}</strong><br>
            <small>🔗 Matched → <em>[{match['matched_resume_section']}]</em> {match['matched_resume_text'][:120]}...</small><br>
            <small>Score: <strong>{score:.3f}</strong></small>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # RAG RETRIEVAL RESULTS
    # ══════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">🗄️ RAG Retrieval (ChromaDB)</div>', unsafe_allow_html=True)
    st.caption("For each JD requirement, top-3 resume chunks retrieved from the vector database.")

    with st.expander("View RAG results", expanded=False):
        for rag in result["rag_results"]:
            st.markdown(f"**[{rag['jd_category']}]** {rag['jd_text'][:100]}...")
            for hit in rag["retrieved_chunks"]:
                st.markdown(f"""
                <div class="rag-card">
                    <strong>[{hit['section']}]</strong> {hit['text'][:150]}... — Score: <strong>{hit['score']:.3f}</strong>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # LLM REASONING
    # ══════════════════════════════════════════════════════════════════

    if result["explanations"]:
        st.markdown('<div class="section-header">🤖 LLM Match Explanations</div>', unsafe_allow_html=True)
        for exp in result["explanations"]:
            with st.expander(
                f"{'✅' if exp['score'] >= 0.5 else '❌'} {exp['jd_text'][:80]}... (score: {exp['score']:.3f})"
            ):
                st.markdown(f"**JD Requirement:** {exp['jd_text']}")
                st.markdown(f"**Resume Section:** {exp['resume_text'][:300]}")
                st.markdown(f"**LLM Analysis:** {exp['explanation']}")

    # ══════════════════════════════════════════════════════════════════
    # IMPROVEMENT SUGGESTIONS
    # ══════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">💡 Improvement Suggestions</div>', unsafe_allow_html=True)

    if result["improvement_suggestions"]:
        st.markdown(result["improvement_suggestions"])
    else:
        st.success("✅ Strong alignment! Consider adding quantified achievements to further strengthen your resume.")