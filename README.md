# RAG Chatbot Tư Vấn Bệnh Nhân Phụ Sản

Hệ thống chatbot **Retrieval-Augmented Generation (RAG)** phục vụ tư vấn bệnh nhân tại bệnh viện phụ sản.
Sử dụng dữ liệu từ các giáo trình y khoa và phác đồ điều trị sản phụ khoa để trả lời câu hỏi một cách chính xác, an toàn và có trích dẫn nguồn.

---

## Kiến trúc Pipeline

### Stage 1 — Indexing (Offline)

```
PDF gốc (data/raw/)
    │
    ▼
[Loader — Docling]       PDF → Markdown chất lượng cao (data/interim/)
    │
    ▼
[Cleaner]                Chuẩn hoá Unicode, sửa lỗi font, xoá mục lục (data/cleaned/)
    │
    ▼
[Chunker]                Chia văn bản: Hierarchical hoặc Semantic
    │
    ▼
[Embedder]               Chuyển chunk → vector (sentence-transformers)
    │
    ▼
[Vector Store]           Lưu vector (ChromaDB / FAISS) — chưa triển khai
```

### Stage 2 — Generation (Real-time)

```
Câu hỏi bệnh nhân
    │
    ▼
[Pre-retrieval]          Làm sạch / mở rộng / viết lại câu hỏi
    │
    ▼
[Retrieval]              Dense search trên Vector Store
    │
    ▼
[Post-retrieval]         Reranking, lọc nhiễu
    │
    ▼
[Generation]             LLM tổng hợp câu trả lời có trích dẫn nguồn
```

---

## Nguồn dữ liệu

| File | Mô tả |
|---|---|
| `Huong dan quoc gia dai thao duong thai ky - San phu khoa.pdf` | Hướng dẫn quốc gia về đái tháo đường thai kỳ |
| `Huong dan quoc gia ve cac dich vu cham soc suc khoe sinh san.pdf` | Hướng dẫn chăm sóc sức khoẻ sinh sản (~1100 trang) |
| `Thuc-hanh-LS-SPK.pdf` | Thực hành lâm sàng Sản phụ khoa |

---

## Kết quả Benchmark (Embedding × Chunking)

Đánh giá trên **30 câu hỏi** (24 in-domain + 6 out-of-domain) với 2 chiến lược chunking.

### Recall@k — Tỷ lệ tìm ra đoạn văn bản đúng trong top-k

| Model | Chunker | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| `BAAI/bge-m3` | hierarchical | 0.500 | 0.542 | 0.625 |
| `BAAI/bge-m3` | semantic | 0.542 | 0.625 | **0.667** |
| `intfloat/multilingual-e5-large` | hierarchical | 0.542 | 0.583 | **0.667** |
| `intfloat/multilingual-e5-large` | semantic | 0.542 | 0.542 | 0.625 |
| `AITeamVN/Vietnamese_Embedding` | hierarchical | 0.500 | 0.625 | **0.667** |
| `AITeamVN/Vietnamese_Embedding` | **semantic** | 0.583 | 0.625 | **0.667** |

### MRR — Chất lượng xếp hạng (đoạn đúng xuất hiện ở vị trí nào)

| Model | Chunker | MRR |
|---|---|---|
| `BAAI/bge-m3` | hierarchical | 0.482 |
| `BAAI/bge-m3` | semantic | 0.494 |
| `intfloat/multilingual-e5-large` | hierarchical | 0.541 |
| `intfloat/multilingual-e5-large` | semantic | 0.480 |
| `AITeamVN/Vietnamese_Embedding` | hierarchical | 0.517 |
| **`AITeamVN/Vietnamese_Embedding`** | **semantic** | **0.578** |

### Out-of-Domain Similarity — Khả năng từ chối câu hỏi ngoài miền

| Model | OOD sim (thấp hơn = tốt hơn) |
|---|---|
| `BAAI/bge-m3` | ~0.50 (tốt) |
| `intfloat/multilingual-e5-large` | ~0.82 (cảnh báo) |
| **`AITeamVN/Vietnamese_Embedding`** | **~0.29** (tốt nhất) |

### Phân tích

- **`AITeamVN/Vietnamese_Embedding` + Semantic Chunking** là tổ hợp tốt nhất toàn diện:
  MRR cao nhất (0.578), Recall@10 đồng đều, OOD sim thấp nhất (0.29) — phân biệt câu hỏi lạ tốt nhất.
- **Semantic Chunking** nhất quán hơn Hierarchical ở MRR — đoạn đúng được xếp hạng cao hơn.
- **`multilingual-e5-large`** có MRR tốt với hierarchical nhưng OOD sim rất cao (~0.82)
  — dễ trả lời nhầm câu hỏi ngoài miền (đặc điểm kiến trúc XLM-RoBERTa, không phải lỗi prefix).

> **Lưu ý phương pháp:** `e5-large` bị giới hạn `max_seq_length=512` (kiến trúc cứng),
> trong khi `bge-m3` và `Vietnamese_Embedding` xử lý đến 4096 token.
> Đây là so sánh **as-designed**, không phải so kiến trúc thuần túy.

---

## Roadmap

| Bước | Module | Trạng thái |
|---|---|---|
| 1 | **Loader** — Docling PDF sang Markdown | Hoàn thành |
| 2 | **Cleaner** — Unicode, font, mục lục | Hoàn thành |
| 3 | **Chunking** — Hierarchical + Semantic | Hoàn thành |
| 4 | **Embedding Eval** — So sánh 3 model x 2 chunker | Hoàn thành |
| 5 | **Vector Store** — ChromaDB / FAISS | Chưa làm |
| 6 | **Retrieval Pipeline** — Dense / Hybrid search | Chưa làm |
| 7 | **Generation** — LLM + Prompt template | Chưa làm |
| 8 | **End-to-End Eval** — RAGAS hoặc tương đương | Chưa làm |
| 9 | **UI** — Streamlit / Gradio / FastAPI | Chưa làm |

---

## Cấu trúc thư mục

```
rag-phu-san-chatbot/
├── README.md
├── requirements.txt            # Thư viện Python (có phiên bản)
├── pyproject.toml              # Package config (editable install)
├── .gitignore
├── .env.example
├── commands_to_run.txt         # Tất cả lệnh có thể chạy trong dự án
│
├── data/
│   ├── raw/                    # PDF gốc (gitignore, chỉ giữ .gitkeep)
│   ├── interim/                # Markdown thô từ Docling (gitignore)
│   └── cleaned/                # Markdown đã làm sạch, sẵn sàng chunk (gitignore)
│
├── src/
│   ├── __init__.py
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── loader.py           # Docling: PDF sang Markdown (data/interim/)
│   │   ├── cleaner.py          # Làm sạch Markdown (data/cleaned/)
│   │   ├── chunking/
│   │   │   ├── base.py         # Interface BaseChunker + Chunk dataclass
│   │   │   ├── config.py       # HierarchicalChunkerConfig, SemanticChunkerConfig
│   │   │   ├── hierarchical_chunker.py
│   │   │   ├── semantic_chunker.py
│   │   │   └── table_utils.py  # Bảo vệ bảng atomic khi chunking
│   │   └── embedding/
│   │       ├── base.py         # Interface BaseEmbedder
│   │       ├── config.py       # EmbedderConfig + MODELS_TO_COMPARE
│   │       └── embedder.py     # SentenceTransformerEmbedder (hỗ trợ E5 prefix)
│   └── generation/
│       └── __init__.py         # Rỗng — sẽ bổ sung (LLM, prompt, retrieval)
│
└── evaluation/
    ├── eval_questions.json     # 30 câu hỏi benchmark (24 in-domain + 6 OOD)
    ├── metrics.py              # Recall@k, Precision@k, MRR
    ├── run_chunking_eval.py    # Chạy chunker, lưu cache JSON
    ├── run_embedding_eval.py   # Benchmark 3 model x 2 chunker, xuất CSV
    └── results/
        ├── chunks_cache/       # hierarchical.json, semantic.json (gitignore)
        └── comparison_matrix.csv  # Kết quả benchmark (gitignore)
```
