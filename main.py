"""
MamaMind AI — main.py  (FastAPI backend)
─────────────────────────────────────────
Setup:
  pip install fastapi uvicorn pymupdf faiss-cpu sentence-transformers
              google-generativeai python-dotenv

Configure .env:
  GOOGLE_API_KEY=your_key_here
  DATA_PATH=C:\\path\\to\\your\\pdfs

Run:
  python main.py
  → opens at  http://localhost:8000
  → open  index.html  in browser and set API to  http://localhost:8000
"""

import numpy as np
import fitz
import os, re, json
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

# ─────────────────────────────────────────────
#  ENV & AI CONFIG
# ─────────────────────────────────────────────
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))

# ─────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────
# NOTE for Render deployment:
#   - Set DATA_PATH env var in Render dashboard to your PDF folder path
#   - Set GOOGLE_API_KEY env var in Render dashboard
#   - Render automatically sets PORT; uvicorn reads it below
#   - index.html must be in the same directory as main.py (monorepo root)
app = FastAPI(title="MamaMind AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve index.html from the same directory as main.py (Render-safe)."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
def health():
    count = faiss_index.ntotal if faiss_index else 0
    return {"status": "ok", "service": "MamaMind AI", "chunks": count}

# ─────────────────────────────────────────────
#  EMBEDDING MODEL (lazy-loaded)
# ─────────────────────────────────────────────
_emb_model = None

def get_emb_model():
    global _emb_model
    if _emb_model is None:
        print("Loading embedding model…")
        _emb_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model ready")
    return _emb_model

# ─────────────────────────────────────────────
#  PDF → CHUNKS → FAISS INDEX
# ─────────────────────────────────────────────

def load_pdf_files(path: str) -> dict:
    files = {}
    if not os.path.isdir(path):
        print(f"⚠️  Directory not found: {path}")
        return files
    for pdf in os.listdir(path):
        if pdf.endswith(".pdf"):
            doc  = fitz.open(os.path.join(path, pdf))
            text = "".join(page.get_text() for page in doc)
            files[pdf] = text
            print(f"  Loaded {pdf} ({len(text):,} chars)")
    return files


def chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    chunks, start = [], 0
    while start <= len(text):
        chunks.append(text[start: start + chunk_size])
        start += chunk_size - overlap
    return chunks


def build_or_load_index(data_path: str):
    idx_f = "faiss_index.index"
    cks_f = "faiss_chunks.json"
    model = get_emb_model()

    if os.path.exists(idx_f) and os.path.exists(cks_f):
        idx = faiss.read_index(idx_f)
        with open(cks_f) as f:
            cks = json.load(f)
        print(f"✅ Loaded FAISS index — {idx.ntotal} chunks")
        return idx, cks

    files = load_pdf_files(data_path)
    if not files:
        print("⚠️  No PDFs found — running in demo mode (no knowledge base)")
        return None, []

    all_chunks = []
    for fname, text in files.items():
        clean = text.replace("\n", " ").strip()
        for ch in chunking(clean):
            all_chunks.append({"file": fname, "content": ch})
        print(f"  Chunked {fname} → {len(all_chunks)} total chunks")

    texts = [c["content"] for c in all_chunks]
    print(f"Generating embeddings for {len(texts)} chunks…")
    embs  = model.encode(texts, batch_size=32, show_progress_bar=True)

    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs.astype("float32"))

    faiss.write_index(idx, idx_f)
    with open(cks_f, "w") as f:
        json.dump(texts, f)

    print(f"✅ FAISS index built — {idx.ntotal} vectors")
    return idx, texts


def vector_search(query: str, idx, chunks: list, k: int = 4) -> list:
    if idx is None:
        return []
    model = get_emb_model()
    q_emb = model.encode([query]).astype("float32")
    _, ii  = idx.search(q_emb, k)
    return [chunks[i] for i in ii[0]]


def safety_filter(text: str) -> str:
    for pat in [
        r"\b\d+\s?(mg|ml)\b",
        r"per\s?kg",
        r"every\s?\d+\s?hours",
        r"\bdose\b",
        r"\bdosage\b",
    ]:
        text = re.sub(pat, "[removed]", text, flags=re.IGNORECASE)
    return text

# ─────────────────────────────────────────────
#  STARTUP — LOAD PDF DATA
# ─────────────────────────────────────────────
DATA_PATH = os.getenv("DATA_PATH", "data")
print(f"\n🌸 MamaMind AI starting up…")
print(f"   PDF data path: {DATA_PATH}")
faiss_index, faiss_chunks = build_or_load_index(DATA_PATH)

# ─────────────────────────────────────────────
#  PYDANTIC MODELS
# ─────────────────────────────────────────────

class Profile(BaseModel):
    name:          Optional[str] = "Friend"
    baby_age:      Optional[str] = "Unknown"
    stage:         Optional[str] = "New Mother"
    delivery_type: Optional[str] = "Not specified"


class ChatMessage(BaseModel):
    role:    str
    content: str


class ChatRequest(BaseModel):
    question: str
    profile:  Optional[Profile] = None
    history:  Optional[List[ChatMessage]] = []


class IndexRequest(BaseModel):
    data_path: str

# ─────────────────────────────────────────────
#  /chat  ENDPOINT
# ─────────────────────────────────────────────

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    question = req.question.strip()
    profile  = req.profile or Profile()
    history  = req.history or []

    ctx = "\n".join(vector_search(question, faiss_index, faiss_chunks))

    profile_text = f"""
- Name:          {profile.name}
- Baby Age:      {profile.baby_age}
- Stage:         {profile.stage}
- Delivery Type: {profile.delivery_type}"""

    history_text = "".join(
        f"{m.role}: {m.content}\n" for m in history[-5:]
    )

    prompt = f"""You are MamaMind AI — a warm, caring assistant for new mothers.

Support with: daily routines, parenting tips, baby care basics (non-medical),
lifestyle management, emotional well-being, time management.

STRICT RULES:
- NEVER provide: medical diagnosis, disease identification, medication advice,
  dosage or treatment, emergency instructions.
- If the question is medical → redirect to lifestyle advice or suggest a professional.

Knowledge Base Context:
{ctx}

User Profile:{profile_text}

Conversation History:
{history_text}

User Question: {question}

INSTRUCTIONS:
- Address the mother by name if available
- Be warm, friendly, and practical
- Keep language simple and supportive
- Avoid technical medical language

RESPONSE FORMAT:
**Answer:**
- Clear explanation

**Practical Tips:**
- Bullet points

**Note:**
- Encouraging closing line
"""

    model_ai = genai.GenerativeModel(
        "gemma-3-27b-it",
        generation_config={"temperature": 0.4, "max_output_tokens": 500},
    )
    raw     = model_ai.generate_content(prompt).text
    cleaned = safety_filter(raw)
    return JSONResponse({"response": cleaned})


# ─────────────────────────────────────────────
#  /index  ENDPOINT — re-index PDFs on demand
# ─────────────────────────────────────────────

@app.post("/index")
async def reindex(req: IndexRequest):
    global faiss_index, faiss_chunks
    for f in ["faiss_index.index", "faiss_chunks.json"]:
        if os.path.exists(f):
            os.remove(f)
    faiss_index, faiss_chunks = build_or_load_index(req.data_path)
    count = faiss_index.ntotal if faiss_index else 0
    return {"status": "ok", "chunks": count, "data_path": req.data_path}


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))   # Render injects PORT automatically
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False)