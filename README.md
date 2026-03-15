# Resume ↔ JD Smart Matcher

A **fully local**, RAG-powered Resume ↔ Job Description matching system using embeddings, ChromaDB, and a lightweight LLM.

No cloud APIs. No data leaves your machine.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-green)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-purple)

---

## Features

- **Resume Upload** — PDF, DOCX, TXT (with OCR fallback for scanned PDFs)
- **JD Input** — paste text or fetch from a public URL
- **Section-Classified Chunking** — resume split into SKILLS, EXPERIENCE, PROJECTS, EDUCATION, etc.
- **ChromaDB Vector Store** — in-memory RAG retrieval per JD requirement
- **Section-wise Full Outer Join** — heatmap grid showing cosine similarity between every resume section × JD category
- **Dynamic Skill Extraction** — KeyBERT-powered, no hardcoded skill lists
- **Hybrid ATS Score** — 50% semantic + 30% skill overlap + 20% requirement coverage (configurable)
- **LLM Reasoning** — Ollama explains each match + generates improvement suggestions
- **Premium Streamlit Dashboard** — gradient score card, color-coded heatmap, skill chips, match cards

---

## Quick Start (Windows)

### Prerequisites

| Tool | Required | Purpose |
|------|----------|---------|
| [Python 3.10+](https://www.python.org/downloads/) | **Yes** | Core runtime. Check "Add to PATH" during install |
| [Ollama](https://ollama.com/download) | Optional | LLM reasoning (explanations & suggestions) |

### Step 1 — Extract the ZIP

Extract the downloaded ZIP to any folder, e.g. `C:\ResumeJDMatcher\`

### Step 2 — Run Setup (one time)

Double-click **`setup.bat`**

This will:
- ✅ Check Python is installed
- ✅ Create a virtual environment (`.venv/`)
- ✅ Install all Python dependencies
- ✅ Pull the Ollama model (if Ollama is installed)

### Step 3 — Launch the App

Double-click **`run.bat`**

This will:
- Start the Ollama server (if installed)
- Launch the Streamlit dashboard
- Auto-open your browser at `http://localhost:8501`

### Step 4 — Use It

1. Upload your resume (PDF / DOCX / TXT)
2. Paste a job description (or enter a public URL)
3. Click **🚀 Analyze Resume**
4. View your match score, section grid, skills, and improvement suggestions

---

## Project Structure

```
Resume-JD-Matcher/
│
├── app/                        # Core modules
│   ├── chunking.py             # Section-classified resume & JD chunking
│   ├── embeddings.py           # BAAI/bge-small-en embedding model
│   ├── vector_store.py         # ChromaDB in-memory vector store
│   ├── similarity.py           # Section-wise full outer join similarity
│   ├── skill_extractor.py      # KeyBERT dynamic skill extraction
│   ├── skill_match.py          # Semantic skill matching with scores
│   └── feedback.py             # Ollama LLM reasoning & suggestions
│
├── data/                       # Sample data for testing
│   ├── sample_resume.txt
│   └── sample_jd.txt
│
├── tesseract/                  # Bundled Tesseract OCR + Poppler
│   ├── tesseract.exe
│   └── poppler/
│
├── main.py                     # Pipeline orchestrator (CLI + importable)
├── app_ui.py                   # Streamlit dashboard UI
├── requirements.txt            # Python dependencies
│
├── setup.bat                   # One-time setup script
├── run.bat                     # Launch the Streamlit app
├── run_cli.bat                 # Run CLI pipeline on sample data
└── README.md                   # This file
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| UI | Streamlit |
| Embeddings | SentenceTransformers (BAAI/bge-small-en) |
| Vector DB | ChromaDB (in-memory) |
| Skill Extraction | KeyBERT |
| LLM | Ollama (qwen2.5:1.5b / phi3:mini / tinyllama) |
| PDF Parsing | pdfplumber + Tesseract OCR |
| DOCX Parsing | python-docx |
| Web Scraping | trafilatura + BeautifulSoup |

---

## Scoring Formula

```
Final Score = 0.50 × Semantic Similarity
            + 0.30 × Skill Overlap
            + 0.20 × Requirement Coverage
```

Weights are adjustable in the sidebar.

| Component | How It's Computed |
|-----------|-------------------|
| Semantic Similarity | Mean of best cosine similarity for each JD requirement |
| Skill Overlap | Fraction of JD skills semantically matched (threshold: 0.65) |
| Requirement Coverage | Fraction of JD requirements with match above 0.50 |

---

## Without Ollama (No LLM)

The app works perfectly without Ollama installed. Toggle off **"Enable LLM Reasoning"** in the sidebar. You'll still get:
- Match scores
- Section mapping grid
- Matched/missing skills
- RAG retrieval results

Only the LLM-powered explanations and improvement suggestions will be disabled.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Python is not installed` | Install from python.org, check "Add to PATH" |
| `setup.bat fails on pip install` | Check internet connection, retry |
| `LLM unavailable` error | Install Ollama + run `ollama pull qwen2.5:1.5b` |
| Scanned PDF gives no text | Tesseract is bundled; ensure the PDF is readable |
| JD URL scraping fails | Some sites block bots. Paste the JD text manually |
| Port 8501 already in use | Close other Streamlit instances, or edit `run.bat` to use another port |

---

