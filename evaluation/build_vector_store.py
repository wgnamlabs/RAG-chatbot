"""
build_vector_store.py — Index toàn bộ corpus vào Qdrant (Docker) và tạo BM25 index.

Bước chuẩn bị — khởi động Qdrant bằng Docker (chạy 1 lần, giữ terminal mở):
    docker run -p 6333:6333 -p 6334:6334 \
        -v D:/rag-phu-san-chatbot/data/vector_db/qdrant:/qdrant/storage \
        qdrant/qdrant

Sau đó chạy script này:
    python evaluation/build_vector_store.py
    python evaluation/build_vector_store.py --chunker hierarchical
    python evaluation/build_vector_store.py --recreate   # xoá collection cũ và build lại
    python evaluation/build_vector_store.py --host localhost --port 6333

Xem vectors tại dashboard: http://localhost:6334/dashboard
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Reconfigure stdout for Windows console
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Đưa src vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indexing.chunking.base import Chunk
from indexing.embedding import SentenceTransformerEmbedder
from indexing.embedding.config import EmbedderConfig
from indexing.vector_store import QdrantVectorStore, QdrantStoreConfig


# ── Cấu hình mặc định ──────────────────────────────────────────────────────────

DEFAULT_CHUNKER   = "semantic"
DEFAULT_EMB_MODEL = "AITeamVN/Vietnamese_Embedding"
DEFAULT_EMB_DIM   = 1024

# ── Tokenizer tiếng Việt cho BM25 ──────────────────────────────────────────────

def _build_vn_tokenizer():
    """Trả về hàm tokenize tiếng Việt. Ưu tiên underthesea, fallback pyvi, fallback split."""
    try:
        from underthesea import word_tokenize
        print("[BM25] Dùng underthesea tokenizer.")
        return lambda text: word_tokenize(text, format="text").split()
    except ImportError:
        pass
    try:
        from pyvi import ViTokenizer
        print("[BM25] underthesea không có, dùng pyvi tokenizer.")
        return lambda text: ViTokenizer.tokenize(text).split()
    except ImportError:
        pass
    print("[BM25] ⚠️  Cả underthesea và pyvi đều không có, dùng split() cơ bản.")
    return lambda text: text.lower().split()


# ── Main ───────────────────────────────────────────────────────────────────────

def build_stores(
    base_path: Path,
    chunker_name: str = DEFAULT_CHUNKER,
    emb_model: str = DEFAULT_EMB_MODEL,
    emb_dim: int = DEFAULT_EMB_DIM,
    recreate: bool = False,
    device: str = "cuda",
    host: str = "localhost",
    port: int = 6333,
) -> None:
    # 1. Load chunks cache
    chunk_file = base_path / "evaluation" / "results" / "chunks_cache" / f"{chunker_name}.json"
    if not chunk_file.exists():
        print(f"❌ Không tìm thấy {chunk_file}. Hãy chạy run_chunking_eval.py trước.")
        sys.exit(1)

    print(f"\n[Step 1] Đọc chunks từ {chunk_file}...")
    with open(chunk_file, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    chunks: List[Chunk] = [
        Chunk(text=c["text"], metadata=c["metadata"])
        for c in raw_chunks
    ]
    print(f"  → {len(chunks)} chunks loaded.")

    # 2. Embed chunks
    emb_config = EmbedderConfig(
        model_name=emb_model,
        batch_size=32,
        max_seq_length=4096,
        device=device,
    )
    embedder = SentenceTransformerEmbedder(emb_config)
    print(f"\n[Step 2] Embedding {len(chunks)} chunks với {emb_model}...")
    embedder.load()
    chunk_texts = [c.text for c in chunks]
    embeddings = embedder.encode(chunk_texts, is_query=False)
    embedder.unload()
    print(f"  → Embeddings shape: {embeddings.shape}")

    # 3. Upsert vào Qdrant (Docker server)
    qdrant_config = QdrantStoreConfig(
        collection_name="phu_san_chunks",
        host=host,
        port=port,
        vector_size=emb_dim,
        distance="cosine",
    )
    store = QdrantVectorStore(config=qdrant_config)
    store.load()
    store.create_collection(recreate=recreate)

    print(f"\n[Step 3] Upsert vào Qdrant Docker server...")
    store.add(chunks, embeddings, chunker_type=chunker_name)

    info = store.collection_info()
    print(f"  → Collection info: {info}")

    # 4. Build BM25 index
    print(f"\n[Step 4] Build BM25 index...")
    from rank_bm25 import BM25Okapi

    tokenize = _build_vn_tokenizer()
    tokenized_corpus = [tokenize(c.text) for c in chunks]

    bm25 = BM25Okapi(tokenized_corpus)

    bm25_dir = base_path / "data" / "vector_db"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = bm25_dir / "bm25_index.pkl"

    bm25_data = {
        "bm25":         bm25,
        "chunks":       [{"text": c.text, "metadata": c.metadata} for c in chunks],
        "chunker_type": chunker_name,
        "emb_model":    emb_model,
    }
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"  → BM25 index saved: {bm25_path}")
    print(f"\n✅ Hoàn thành! Qdrant: {host}:{port} (collection: {qdrant_config.collection_name}), BM25: {bm25_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Qdrant (Docker) + BM25 index.")
    parser.add_argument("--chunker",  default=DEFAULT_CHUNKER,   help="semantic | hierarchical")
    parser.add_argument("--model",    default=DEFAULT_EMB_MODEL, help="HuggingFace embedding model name")
    parser.add_argument("--dim",      default=DEFAULT_EMB_DIM,   type=int, help="Embedding dimension")
    parser.add_argument("--device",   default="cuda",            help="cuda | cpu")
    parser.add_argument("--recreate", action="store_true",       help="Xoá và tạo lại collection Qdrant")
    parser.add_argument("--host",     default="localhost",       help="Qdrant server host")
    parser.add_argument("--port",     default=6333, type=int,   help="Qdrant REST API port")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent
    build_stores(
        base_path=base_path,
        chunker_name=args.chunker,
        emb_model=args.model,
        emb_dim=args.dim,
        recreate=args.recreate,
        device=args.device,
        host=args.host,
        port=args.port,
    )
