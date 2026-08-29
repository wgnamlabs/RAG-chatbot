"""
web_app.py -- FastAPI server cho giao diện web Chatbot Me Bau.

Chay:
    python web_app.py
    Mo trinh duyet tai: http://localhost:8000

Yeu cau: pip install fastapi uvicorn
Pipeline va Qdrant phai dang chay (giong nhu ask.py).
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import pickle
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from indexing.embedding import SentenceTransformerEmbedder
from indexing.embedding.config import EmbedderConfig
from indexing.vector_store import QdrantVectorStore, QdrantStoreConfig
from generation.pipeline.main import run_pipeline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Khởi tạo infra một lần duy nhất (giống ask.py) ──────────────────────────
print("[web_app] Dang tai mo hinh embedding va ket noi database...")


embedder = SentenceTransformerEmbedder(EmbedderConfig(
    model_name="AITeamVN/Vietnamese_Embedding_v2",
    batch_size=32,
    max_seq_length=2048,
    device="cuda",
))
embedder.load()

qdrant = QdrantVectorStore(config=QdrantStoreConfig(collection_name="phu_san_chunks"))
qdrant.load()

with open("data/vector_db/bm25_index.pkl", "rb") as f:
    bm25_data = pickle.load(f)

print("[web_app] San sang! Truy cap http://localhost:8000")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Chatbot Tư Vấn Sức Khỏe Mẹ Bầu", version="1.0.0")

# Serve static files (index.html + assets)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency: dict


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve giao diện chính."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Chưa có file static/index.html</h1>", status_code=404)


@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import FileResponse
    return FileResponse(STATIC_DIR / "favicon.ico")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Endpoint nhận câu hỏi, trả về câu trả lời từ RAG pipeline."""
    query = req.message.strip()
    if not query:
        return ChatResponse(answer="Mẹ vui lòng nhập câu hỏi nhé! 💕", sources=[], latency={})

    result = run_pipeline(
        query=query,
        qdrant_store=qdrant,
        bm25_data=bm25_data,
        embedder=embedder,
    )

    # Deduplicate sources (giống ask.py)
    seen = set()
    sources_list = []
    for s in (result.sources_used or []):
        key = f"{s.source} § {s.section or 'N/A'}"
        if key not in seen:
            seen.add(key)
            sources_list.append({"source": s.source, "section": s.section or ""})

    return ChatResponse(
        answer=result.answer,
        sources=sources_list,
        latency=result.latency_breakdown or {},
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "qwen3.5:4b"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
