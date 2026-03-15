"""Quick test for JD noise filtering."""
from app.chunking import smart_chunk_jd, _is_jd_noise

# Test noise lines
noise_lines = [
    "PG: Any Postgraduate",
    "UG: Any Graduate",
    "Role: Back End Developer",
    "Role Category: Software Development",
    "Industry: IT",
    "Department: Engineering",
    "Location: Bangalore",
    "Salary: 10-15 LPA",
    "Experience: 3-5 years",
    "Key Skills:",
    "Employment Type: Full Time",
]

print("=== NOISE DETECTION ===")
for line in noise_lines:
    result = _is_jd_noise(line)
    status = "FILTERED" if result else "KEPT"
    print(f"  [{status}] {line}")

# Test full JD chunking
test_jd = """Job description
Design and deliver AI/ML and GenAI solutions from discovery to production.
Build models for predictive analytics and time-series forecasting.
Role: Back End Developer
Role Category: Software Development
Industry: IT Services
Department: Engineering
PG: Any Postgraduate
UG: Any Graduate
Location: Bangalore
Salary: 10-15 LPA
Experience: 3-5 years
Key Skills:
Advanced Python Azure MLOps GCP LLMOps Cloud Generative AI
Requirements
Experience with Python, AWS, and ML deployment required.
Strong knowledge of deep learning frameworks like TensorFlow or PyTorch.
Skills Required
Proficiency in cloud platforms (AWS, GCP, Azure).
Experience building and deploying ML pipelines."""

print("\n=== JD CHUNKS ===")
chunks = smart_chunk_jd(test_jd)
print(f"Total chunks: {len(chunks)}")
for c in chunks:
    print(f"  [{c['category']}] {c['text'][:100]}")
