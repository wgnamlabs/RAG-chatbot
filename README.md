# RAG Chatbot Tư Vấn Phụ Sản

Hệ thống chatbot **Retrieval-Augmented Generation (RAG)** tư vấn sức khỏe sản phụ khoa bằng tiếng Việt. Toàn bộ LLM (rewrite + generation) chạy **local qua Ollama** — không gọi bất kỳ cloud API nào.

---

## Kiến trúc Pipeline

### Stage 1 — Indexing (Offline)

```
data/raw/*.pdf
    │
    ▼ loader.py (Docling — PDF → Markdown)
data/interim/*.md
    │
    ▼ cleaner.py (chuẩn hoá Unicode, xóa lỗi font)
data/cleaned/*.md
    │
    ▼ chunking/ (Semantic Chunking)
    │
    ├─▶ embedding/ (AITeamVN/Vietnamese_Embedding_v2, dim=1024)
    │       └─▶ Qdrant vector store (localhost:6333)
    │
    └─▶ BM25 index (underthesea tokenizer)
            └─▶ data/vector_db/bm25_index.pkl
```

### Stage 2 — Generation Pipeline (Online)

```
Câu hỏi người dùng
    │
    ▼ rewrite.py       — Query rewriting (qwen3:4b, Ollama local)
    │
    ▼ retrieval.py     — Hybrid RRF (Dense + BM25)
    │                    top_k=15, rrf_k=60
    ▼ postprocess.py   — dedup_redundant + sandwich_order
    │
    ▼ prompt.py        — Build prompt (section citation thay trang)
    │
    ▼ main.py          — Generate (qwen3:8b, Ollama local, temperature=0)
    │
    ▼ PipelineOutput (answer + sources_used + latency_breakdown)
```

---

## CHANGELOG / Quyết định kỹ thuật

### 2024-08 — Chốt cấu hình retrieval: **Hybrid RRF** (bỏ rerank)

Eval trên 127 câu in-domain + 24 câu OOD (Vietnamese medical corpus):

| Cấu hình | R@10 | P@5 | MRR | Latency P95 |
|----------|------|-----|-----|-------------|
| Dense only | 0.9722 | 0.7000 | 0.9028 | 0.069s |
| **Hybrid RRF** ✅ | **0.9861** | **0.7556** | **0.9167** | **0.140s** |
| Hybrid + Rerank | 0.9861 | 0.7111 | 0.9028 | 3.454s |

**Lý do bỏ rerank** (`use_rerank=False` trong DEFAULT_CONFIG):
- MRR giảm: 0.9167 → 0.9028
- Precision@5 giảm: 0.7556 → 0.7111
- Latency P95 tăng: 0.14s → 3.45s (**~25×**)
- Debug: `bge-reranker-v2-m3` mismatch văn phong protocol lâm sàng tiếng Việt

**Để thử reranker mới:** set `use_rerank=True` trong config. Code path vẫn còn trong `src/generation/pipeline/main.py` và `rerank.py`.

---

## Cấu trúc thư mục

```
rag-phu-san-chatbot/
├── data/
│   ├── raw/                    # PDF gốc (không push — tự chuẩn bị)
│   ├── interim/                # Markdown sau Docling (không push)
│   ├── cleaned/                # Markdown sau cleaner (không push)
│   └── vector_db/              # Qdrant + BM25 (không push — build lại)
│
├── src/
│   ├── indexing/
│   │   ├── loader.py           # PDF → Markdown (Docling)
│   │   ├── cleaner.py          # Chuẩn hoá văn bản
│   │   ├── chunking/           # Semantic chunking
│   │   ├── embedding/          # Vietnamese_Embedding_v2
│   │   └── vector_store/       # Qdrant interface
│   │
│   └── generation/
│       ├── schemas.py          # Chunk, PipelineOutput (Pydantic)
│       └── pipeline/
│           ├── rewrite.py      # Query rewriting (qwen3:4b)
│           ├── retrieval.py    # Dense + BM25 + Hybrid RRF
│           ├── rerank.py       # CrossEncoder (DEPRECATED, use_rerank=False)
│           ├── postprocess.py  # dedup_redundant + sandwich_order
│           ├── prompt.py       # Build prompt
│           ├── main.py         # run_pipeline() — entry point
│           └── system_prompt.txt
│
├── evaluation/
│   ├── eval_questions.json     # 151 câu benchmark (127 in-domain, 24 OOD)
│   ├── metrics.py              # Recall@k, Precision@k, MRR helpers
│   ├── build_vector_store.py   # Build Qdrant + BM25 từ cleaned/
│   ├── run_embedding_eval.py   # Eval embedding retrieval
│   ├── run_chunking_eval.py    # So sánh chunking strategies
│   ├── run_bm25_eval.py        # Eval BM25
│   ├── eval_pipeline.py        # So sánh 3 cấu hình (dense/hybrid/rerank)
│   ├── faithfulness_eval.py    # LLM-as-judge (qwen3:8b)
│   ├── escalation_eval.py      # Kiểm tra hành vi câu khẩn cấp
│   ├── frr_eval.py             # False Rejection Rate
│   ├── debug_rerank.py         # Phân tích rank shift của reranker
│   └── results/
│       └── chunks_cache/       # Kết quả chunking eval (hierarchical/semantic)
│
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Cài đặt

### Yêu cầu

- Python 3.10+
- Docker (chạy Qdrant)
- [Ollama](https://ollama.ai/) (chạy LLM local)
- GPU NVIDIA (khuyến nghị, CPU cũng chạy được nhưng chậm hơn)

### Bước 1 — Clone & cài thư viện

```bash
git clone https://github.com/<your-repo>/rag-phu-san-chatbot.git
cd rag-phu-san-chatbot

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
pip install -e .
```

### Bước 2 — Chuẩn bị dữ liệu

Đặt 3 file PDF vào `data/raw/`:
- `Huong dan quoc gia dai thao duong thai ky - Sản phụ khoa.pdf`
- `Hướng dẫn quốc gia về các dịch vụ chăm sóc sức khỏe sinh sản.pdf`
- `Thuc-hanh-LS-SPK.pdf`

### Bước 3 — Khởi động hạ tầng

```bash
# Qdrant vector database
docker run -d -p 6333:6333 -p 6334:6334 \
  -v D:/rag-phu-san-chatbot/data/vector_db/qdrant:/qdrant/storage \
  --name qdrant-phu-san qdrant/qdrant

# Ollama models
ollama pull qwen3:4b    # query rewriting
ollama pull qwen3:8b    # generation
```

### Bước 4 — Build index

```bash
# Chạy loader (PDF → Markdown)
python src/indexing/loader.py

# Chạy cleaner (chuẩn hoá văn bản)
python src/indexing/cleaner.py

# Build vector store (Qdrant + BM25)
python evaluation/build_vector_store.py
```

---

## Chạy pipeline

```python
import pickle
from src.indexing.embedding import SentenceTransformerEmbedder
from src.indexing.embedding.config import EmbedderConfig
from src.indexing.vector_store import QdrantVectorStore, QdrantStoreConfig
from src.generation.pipeline.main import run_pipeline

# Load infra (1 lần)
embedder = SentenceTransformerEmbedder(EmbedderConfig(
    model_name="AITeamVN/Vietnamese_Embedding_v2", device="cuda"
))
embedder.load()

qdrant = QdrantVectorStore(config=QdrantStoreConfig(collection_name="phu_san_chunks"))
qdrant.load()

with open("data/vector_db/bm25_index.pkl", "rb") as f:
    bm25_data = pickle.load(f)

# Chạy pipeline
result = run_pipeline(
    query="Tiêu chuẩn chẩn đoán đái tháo đường thai kỳ là gì?",
    qdrant_store=qdrant,
    bm25_data=bm25_data,
    embedder=embedder,
)

print(result.answer)
print("Latency:", result.latency_breakdown)
```

---

## Evaluation

```bash
# Eval retrieval (embedding quality)
python evaluation/run_embedding_eval.py

# So sánh 3 cấu hình retrieval
python evaluation/eval_pipeline.py --device cuda

# Faithfulness (LLM-as-judge, cần Ollama)
python evaluation/faithfulness_eval.py --n 50 --save-cases --device cuda

# Escalation (hành vi câu khẩn cấp)
python evaluation/escalation_eval.py --device cuda

# False Rejection Rate
python evaluation/frr_eval.py --device cuda
```

---

## Tập dữ liệu đánh giá

`evaluation/eval_questions.json` — **151 câu** được phân bổ:

| Tài liệu | Tag | Số câu |
|----------|-----|--------|
| Hướng dẫn quốc gia SKSS | `skss_quoc_gia` | 49 |
| Thực hành lâm sàng SPK | `thuc_hanh_spk` | 41 |
| Đái tháo đường thai kỳ | `dtd_thai_ky` | 37 |
| Out-of-domain | `out_of_domain` | 24 |

Mỗi câu có 5 loại: `direct`, `paraphrase`, `multi_hop`, `applied`, `boundary`.

---

## Stack kỹ thuật

| Component | Công nghệ |
|-----------|-----------|
| PDF parsing | [Docling](https://github.com/DS4SD/docling) |
| Embedding | [AITeamVN/Vietnamese_Embedding_v2](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2) (dim=1024) |
| Vector DB | [Qdrant](https://qdrant.tech/) |
| Sparse retrieval | BM25 + underthesea tokenizer |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| LLM | [Ollama](https://ollama.ai/) — qwen3:4b (rewrite), qwen3:8b (generation) |
| Schema | Pydantic v2 |