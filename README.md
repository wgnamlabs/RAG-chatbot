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

### Chọn cấu hình embedding: AITeamVN/Vietnamese_Embedding_v2 + Semantic Chunking

So sánh dense-only (`run_embedding_eval.py` + `run_chunking_eval.py`) giữa 3 embedding model × 2 chiến lược chunking, ở k=10:

| Chunker | Embedding | Recall@10 | Precision@10 | MRR | Top1-sim OOD |
|---|---|---|---|---|---|
| **semantic** ✅ | **AITeamVN/Vietnamese_Embedding_v2** | 0.9846 | 0.7162 | **0.9228** | **0.3619** |
| hierarchical | AITeamVN/Vietnamese_Embedding_v2 | 0.9846 | 0.7015 | 0.9024 | 0.3634 |
| semantic | Qwen/Qwen3-Embedding-4B | 0.9923 | 0.7092 | 0.9177 | 0.3801 |
| hierarchical | Qwen/Qwen3-Embedding-4B | 1.0000 | 0.7123 | 0.9224 | 0.3862 |
| semantic | BAAI/bge-m3 | 0.9769 | 0.7085 | 0.8865 | 0.5011 |
| hierarchical | BAAI/bge-m3 | 0.9923 | 0.7200 | 0.8946 | 0.5001 |

**Lý do chọn AITeamVN/Vietnamese_Embedding_v2 + semantic chunking:**

- **MRR cao nhất** trong 6 tổ hợp (0.9228) — chunk đúng thường nằm ở vị trí top-1/top-2 hơn các cấu hình còn lại, quan trọng vì pipeline chỉ lấy top_k=15 rồi qua RRF chứ không rerank.
- **Top1-sim OOD thấp nhất** (0.3619) — với câu hỏi ngoài phạm vi 3 tài liệu nguồn, model này cho similarity thấp nhất, tức ít khả năng "tự tin nhầm" và kéo theo context sai vào prompt. Đây là tiêu chí quan trọng hơn recall thuần trong bài toán tư vấn y tế, vì retrieve sai cho câu ngoài domain có thể khiến LLM generate câu trả lời sai lệch thay vì từ chối.
- Chênh lệch recall so với Qwen3-Embedding-4B (0.9846 vs 0.9923–1.0000) là không đáng kể (~1%) và không đủ bù lại việc OOD sim cao hơn ~5%.
- BAAI/bge-m3 tuy precision cạnh tranh nhưng OOD sim cao hơn hẳn (~0.50, gần gấp rưỡi) — rủi ro cao nhất trong 3 model, loại khỏi lựa chọn.
- Giữa 2 chiến lược chunking trên cùng embedding AITeamVN, semantic nhỉnh hơn hierarchical ở cả precision (0.7162 vs 0.7015) lẫn MRR (0.9228 vs 0.9024), OOD sim gần như tương đương.

### Chọn cấu hình retrieval: Hybrid RRF (bỏ rerank)

So sánh 3 cấu hình retrieval trên embedding + chunking đã chọn ở trên (`eval_pipeline.py`):

| Cấu hình | R@5 | R@10 | P@5 | P@10 | MRR | Latency (Avg / P50 / P95) |
|----------|-----|------|-----|------|-----|---------------------------|
| Dense only | 0.9846 | 0.9846 | 0.7277 | 0.7162 | 0.9269 | 0.060s / 0.053s / 0.079s |
| **Hybrid RRF** ✅ | **0.9923** | **0.9923** | **0.7692** | 0.7008 | 0.9269 | 0.091s / 0.090s / 0.113s |
| Hybrid + Rerank | 0.9923 | 0.9923 | 0.7308 | 0.7091 | 0.9308 | 2.642s / 2.115s / 4.351s |

**Out-of-Domain — khả năng từ chối câu hỏi ngoài miền:** 30/30 (100%) — không có case nào retrieve nhầm context và trả lời sai cho câu hỏi ngoài phạm vi 3 tài liệu nguồn.

**Lý do chọn Hybrid RRF:**

- **P@5 tốt nhất** trong 3 cấu hình (0.7692), cao hơn Dense only (0.7277) và Hybrid + Rerank (0.7308).
- **Rerank không đáng đánh đổi**: MRR chỉ nhích thêm 0.0039 (0.9308 vs 0.9269) nhưng latency trung bình tăng gấp **~29 lần** (2.642s vs 0.091s), P95 lên tới 4.351s — không phù hợp với chatbot cần phản hồi gần real-time. `use_rerank=False` trong `DEFAULT_CONFIG`; code path vẫn còn trong `src/generation/pipeline/main.py` và `rerank.py` để bật lại khi cần (`use_rerank=True`).
- **Dense only** nhanh hơn (~30ms) nhưng P@5 thấp hơn rõ rệt — mức chênh lệch latency này không đáng để đổi lấy precision thấp hơn.
- **OOD rejection đạt 100%** trên cấu hình Hybrid RRF — xác nhận thêm rằng lựa chọn embedding (AITeamVN, top1-sim OOD thấp nhất) đang phát huy đúng vai trò an toàn khi triển khai thực tế.

> **Đã đối chiếu lại với `eval_questions.json` thật:** bộ câu hỏi hiện có **160 câu tổng — 130 in-domain, 30 OOD** (trong đó 30 OOD gồm 26 `direct` + 4 `ood_boundary`). Số này khớp chính xác với kết quả `eval_pipeline.py` (30/30 OOD refused đúng). Con số "151 câu (127 in-domain, 24 OOD)" ở các chỗ khác trong README trước đây là số liệu cũ — đã cập nhật thống nhất thành 160/130/30 trong toàn bộ tài liệu này.

### Tích hợp Lớp An Toàn Cứng (Safety Guard) và Mở Rộng Thuật Ngữ (Colloquial Terms)

Nhằm đảm bảo an toàn y tế tuyệt đối và tăng cường độ chính xác khi tìm kiếm:
- **`safety_guard.py`**: Lớp bảo vệ cứng (hard-coded rules) độc lập. Phát hiện sớm các câu hỏi chứa dấu hiệu nguy hiểm khẩn cấp (kể cả dùng từ dân dã), chặn các trường hợp tự mâu thuẫn (LLM đưa ra thông tin không nhất quán), và chèn khuyến cáo y tế khẩn cấp nếu câu trả lời chưa đủ độ dứt khoát.
- **`expand_colloquial_terms()` trong `rewrite.py`**: Bổ sung thuật ngữ y khoa chuẩn xác bên cạnh các từ vựng dân dã/địa phương của người dùng một cách deterministic trước khi đưa vào retrieval, giảm thiểu tình trạng miss tài liệu do lệch từ vựng.
- Cập nhật lại `system_prompt.txt` với quy tắc khẩn cấp chặt chẽ hơn, giọng điệu ấm áp và bắt buộc trích nguồn rõ ràng. Toàn bộ được gọi đồng bộ trong `main.py`.


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
│           ├── rewrite.py      # Query rewriting + expand_colloquial_terms()
│           ├── retrieval.py    # Dense + BM25 + Hybrid RRF
│           ├── rerank.py       # CrossEncoder (DEPRECATED, use_rerank=False)
│           ├── postprocess.py  # dedup_redundant + sandwich_order
│           ├── prompt.py       # Build prompt
│           ├── system_prompt.txt # Prompt hệ thống, giọng điệu và nguyên tắc
│           ├── safety_guard.py # Lớp an toàn cứng: danger guard + adversarial guard (kê đơn/chẩn đoán)
│           └── main.py         # entry point ghép safety_guard + expand_colloquial
│
├── evaluation/
│   ├── eval_questions.json     # 160 câu benchmark (130 in-domain, 30 OOD)
│   ├── metrics.py              # Recall@k, Precision@k, MRR helpers
│   ├── build_vector_store.py   # Build Qdrant + BM25 từ cleaned/
│   ├── run_embedding_eval.py   # Eval embedding retrieval
│   ├── run_chunking_eval.py    # So sánh chunking strategies
│   ├── run_bm25_eval.py        # Eval BM25
│   ├── eval_pipeline.py        # So sánh 3 cấu hình (dense/hybrid/rerank)
│   └── results/
│       ├── results_comparison.md
│       └── chunks_cache/       # Kết quả chunking eval (hierarchical/semantic)
│
├── ask.py                      # CLI entrypoint thử chatbot
├── README.md
├── requirements.txt
├── pyproject.toml
└── .env.example
```