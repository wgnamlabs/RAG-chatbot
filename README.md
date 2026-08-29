# RAG Chatbot Tư Vấn Phụ Sản

Hệ thống chatbot **Retrieval-Augmented Generation (RAG)** hỗ trợ tra cứu và tư vấn thông tin sức khỏe sản phụ khoa bằng tiếng Việt.

Pipeline được thiết kế theo hướng chạy local:

- Query rewriting qua **Ollama**
- Dense retrieval + BM25
- Hybrid fusion bằng **Reciprocal Rank Fusion (RRF)**
- Sinh câu trả lời qua **Ollama**
- Trả về câu trả lời kèm nguồn tham chiếu và thông tin latency

> Các tài liệu nguồn, dữ liệu trung gian, vector database và các artifact sinh ra trong quá trình indexing không được lưu trên GitHub.

---

## Kiến trúc hệ thống

### 1. Indexing Pipeline — Offline

```text
Tài liệu đã được tiền xử lý
        │
        ▼
Hierarchical Chunking
        │
        ├───────────────┐
        │               │
        ▼               ▼
Qwen3-Embedding-4B     BM25
        │               │
        ▼               ▼
Qdrant Vector Store   BM25 Index
```

Cấu hình dense retrieval chính thức được chọn sau benchmark:

```text
Chunking  : Hierarchical
Embedding : Qwen/Qwen3-Embedding-4B
Similarity: Cosine similarity
```

Với Qwen3-Embedding-4B:

- Query được encode với `prompt_name="query"`.
- Document/chunk được encode không dùng query prompt.

---

### 2. Online RAG Pipeline

```text
Câu hỏi người dùng
        │
        ▼
Query Rewriting
qwen3:4b — Ollama
        │
        ▼
Hybrid Retrieval
Dense + BM25
        │
        ▼
Reciprocal Rank Fusion
RRF
        │
        ▼
Top candidate chunks
        │
        ▼
Post-processing
dedup_redundant + sandwich_order
        │
        ▼
Prompt Construction
context + citation metadata
        │
        ▼
Generation
qwen3:8b — Ollama
temperature = 0
        │
        ▼
PipelineOutput
answer + sources_used + latency_breakdown
```

Cấu hình retrieval hiện tại:

```text
candidate top_k = 15
rrf_k           = 60
```

`top_k=15` cũng phù hợp với kết quả dense retrieval benchmark: cấu hình được chọn đạt đầy đủ gold evidence trên TEST tại `k=15`.

---

## Các thành phần chính

```text
src/
└── indexing/
    ├── chunking/
    │   ├── base.py
    │   ├── config.py
    │   ├── markdown_utils.py
    │   ├── table_utils.py
    │   ├── hierarchical_chunker.py
    │   └── semantic_chunker.py
    │
    └── embedding/
        ├── base.py
        ├── config.py
        └── embedder.py

evaluation/
├── eval_questions.json
├── eval_questions_manifest.json
├── EVAL_SCHEMA.md
├── metrics.py
├── run_chunking_eval.py
├── run_embedding_eval.py
└── results/
    ├── chunks_cache/
    ├── comparison_matrix_dev.csv
    ├── comparison_matrix_test.csv
    └── embedding_eval/
        ├── per_question_results_dev.csv
        ├── per_question_results_test.csv
        ├── per_group_results_dev.csv
        ├── per_group_results_test.csv
        ├── embedding_summary_dev.json
        ├── embedding_summary_test.json
        ├── evidence_mapping_report_dev.json
        └── evidence_mapping_report_test.json
```

Các file kết quả evaluation có thể được giữ local và không bắt buộc commit lên repository.

---

# Chunking & Embedding Benchmark

## Mục tiêu

Benchmark được dùng để lựa chọn:

- chiến lược chunking;
- embedding model;
- kích thước candidate pool hợp lý cho retrieval.

Hai chiến lược chunking được so sánh:

- `Semantic Chunking`
- `Hierarchical Chunking`

Ba embedding model được đánh giá:

- `AITeamVN/Vietnamese_Embedding_v2`
- `Qwen/Qwen3-Embedding-4B`
- `BAAI/bge-m3`

Benchmark này đánh giá **dense retrieval độc lập**.

Kết quả ở phần này **không phải kết quả cuối của toàn bộ Hybrid RAG pipeline**, vì BM25, RRF, query rewriting và generation chưa được tính vào các metric dưới đây.

---

## Evaluation Protocol

Benchmark sử dụng hai split:

```text
DEV  : 60 queries
TEST : 160 queries
```

TEST gồm:

```text
140 in-domain queries
20  out-of-domain queries
```

Ground truth được định nghĩa ở mức **gold evidence**, không chỉ kiểm tra retrieved chunk có thuộc đúng tài liệu hay không.

Gold evidence được map độc lập sang output của từng chunker trước khi tính metric.

Ở TEST:

```text
Gold evidence units: 164

Semantic:
  mapped   = 164
  unmapped = 0

Hierarchical:
  mapped   = 164
  unmapped = 0
```

---

## Metrics

Các metric chính:

| Metric | Ý nghĩa |
|---|---|
| **Evidence Recall@K** | Tỷ lệ gold evidence units được tìm thấy trong Top-K |
| **Hit@K** | Tỷ lệ query có ít nhất một relevant chunk trong Top-K |
| **Evidence Complete@K** | Tỷ lệ query lấy đủ toàn bộ evidence cần thiết |
| **MRR** | Đánh giá vị trí của relevant result đầu tiên |
| **nDCG@K** | Đánh giá chất lượng thứ hạng của các relevant chunks |
| **OOD AUROC** | Khả năng phân biệt query in-domain và out-of-domain |

Trong quá trình chọn cấu hình, ưu tiên:

```text
Evidence Recall / Evidence Completeness
                ↓
        Ranking Quality
          MRR / nDCG
```

`Precision@K` vẫn được ghi nhận nhưng không dùng làm tiêu chí chính khi so sánh hai chunker vì kích thước và số lượng relevant chunks có thể khác nhau theo cách chia chunk.

---

# Model Selection

## DEV Result

Cấu hình được chọn **trước khi mở TEST**:

```text
Chunking  : Hierarchical
Embedding : Qwen/Qwen3-Embedding-4B
```

Kết quả DEV của cấu hình này:

| Metric | DEV |
|---|---:|
| Evidence Recall@10 | **1.0000** |
| Hit@10 | **1.0000** |
| Evidence Complete@10 | **1.0000** |
| nDCG@10 | **0.8961** |
| MRR | **0.8773** |
| OOD AUROC | **1.0000** |

Sau khi chốt cấu hình trên DEV, TEST chỉ được sử dụng để đánh giá khả năng tổng quát hóa và **không dùng để chọn lại model**.

---

# Final TEST Results

## So sánh 2 chunker × 3 embedding models

Dense retrieval tại `k=10`:

| Chunker | Embedding | Recall@10 | Complete@10 | nDCG@10 | MRR | OOD AUROC |
|---|---|---:|---:|---:|---:|---:|
| Semantic | AITeamVN/Vietnamese_Embedding_v2 | 0.9679 | 0.9429 | 0.7297 | 0.7091 | 0.9657 |
| Hierarchical | AITeamVN/Vietnamese_Embedding_v2 | 0.9679 | 0.9429 | 0.7430 | 0.7062 | 0.9757 |
| Semantic | Qwen/Qwen3-Embedding-4B | 0.9786 | 0.9714 | 0.8386 | 0.8320 | 1.0000 |
| **Hierarchical** | **Qwen/Qwen3-Embedding-4B** | **0.9786** | **0.9786** | **0.8570** | **0.8434** | **1.0000** |
| Semantic | BAAI/bge-m3 | 0.9857 | 0.9786 | 0.7761 | 0.7557 | 0.9996 |
| Hierarchical | BAAI/bge-m3 | **0.9929** | **0.9857** | 0.7962 | 0.7650 | 0.9996 |

### Kết luận

`BAAI/bge-m3 + Hierarchical` đạt Recall@10 cao nhất trên TEST.

Tuy nhiên, cấu hình chính thức vẫn là:

```text
Hierarchical Chunking
+
Qwen/Qwen3-Embedding-4B
```

vì cấu hình này đã được lựa chọn trên DEV trước khi mở TEST, đồng thời cho chất lượng ranking tốt hơn đáng kể trên TEST:

```text
Qwen3 + Hierarchical
nDCG@10 = 0.8570
MRR     = 0.8434

BGE-M3 + Hierarchical
nDCG@10 = 0.7962
MRR     = 0.7650
```

Việc không thay đổi model sau khi xem TEST giúp tránh sử dụng TEST như một tập tuning.

---

## DEV → TEST Generalization

Kết quả của cấu hình đã chọn:

| Metric | DEV | TEST |
|---|---:|---:|
| Evidence Recall@10 | 1.0000 | 0.9786 |
| Evidence Complete@10 | 1.0000 | 0.9786 |
| nDCG@10 | 0.8961 | 0.8570 |
| MRR | 0.8773 | 0.8434 |
| OOD AUROC | 1.0000 | 1.0000 |

Hiệu năng giảm nhẹ từ DEV sang TEST nhưng vẫn duy trì mức retrieval cao, cho thấy cấu hình được chọn tổng quát hóa tốt trên tập đánh giá giữ lại.

---

# Retrieval Analysis

## Recall theo K

Với `Hierarchical + Qwen3-Embedding-4B` trên TEST:

```text
Recall@3  = 0.9000
Recall@5  = 0.9464
Recall@10 = 0.9786
Recall@15 = 1.0000
```

Tại `k=15`, toàn bộ gold evidence trong TEST được phủ:

```text
Evidence Recall@15   = 1.0000
Evidence Complete@15 = 1.0000
Hit@15               = 1.0000
```

Đây là một lý do để sử dụng candidate pool `top_k=15` trước các bước hybrid fusion / post-processing ở pipeline online.

---

## Hard / Multi-evidence Queries

TEST có 24 câu hard/multi-evidence.

Với `Hierarchical + Qwen3-Embedding-4B`:

| K | Evidence Recall | Evidence Complete |
|---:|---:|---:|
| 3 | 0.7083 | 0.5417 |
| 5 | 0.8542 | 0.7500 |
| 10 | 0.9583 | 0.9583 |
| 15 | **1.0000** | **1.0000** |

Các query cần nhiều evidence khó lấy đủ ở Top-3/Top-5, nhưng toàn bộ evidence được retrieve khi mở rộng candidate pool lên Top-15.

---

## Table Queries

TEST có 20 query yêu cầu retrieve nội dung bảng.

Với cấu hình đã chọn:

| K | Recall |
|---:|---:|
| 3 | 0.9500 |
| 5 | **1.0000** |
| 10 | **1.0000** |

Hierarchical chunking vẫn giữ được hiệu quả tốt với các nội dung dạng bảng.

---

## Patient-style Queries

Với 80 patient-style queries trong TEST:

| K | Recall |
|---:|---:|
| 3 | 0.9250 |
| 5 | 0.9500 |
| 10 | 0.9750 |
| 15 | **1.0000** |

Các câu hỏi tự nhiên theo cách người dùng hỏi vẫn duy trì retrieval performance cao.

---

## Numeric Queries

Với 13 numeric queries:

| K | Recall |
|---:|---:|
| 3 | 0.9231 |
| 5 | 0.9231 |
| 10 | 0.9231 |
| 15 | **1.0000** |

Các evidence chứa số liệu khó hơn một chút ở Top-10, nhưng được phủ đầy đủ ở Top-15.

---

# Chạy Evaluation

## DEV

Dùng DEV để thử nghiệm và lựa chọn cấu hình:

```bash
python evaluation/run_embedding_eval.py --split dev
```

## TEST

Chỉ chạy TEST sau khi đã chốt cấu hình trên DEV:

```bash
python evaluation/run_embedding_eval.py --split test
```

Các output chính:

```text
comparison_matrix_<split>.csv
embedding_eval/
├── per_question_results_<split>.csv
├── per_group_results_<split>.csv
├── embedding_summary_<split>.json
└── evidence_mapping_report_<split>.json
```

---

# Quyết định kỹ thuật hiện tại

## Chunking

**Selected: Hierarchical Chunking**

Lý do:

- Giữ được cấu trúc heading/breadcrumb của tài liệu.
- Cho ranking quality tốt hơn Semantic khi kết hợp với Qwen3-Embedding-4B.
- Hoạt động tốt với table queries.
- Đạt full evidence coverage ở `k=15` trên TEST.

---

## Embedding

**Selected: `Qwen/Qwen3-Embedding-4B`**

Lý do:

- Được chọn từ DEV thay vì chọn theo TEST.
- Ranking quality cao nhất trong cấu hình Hierarchical:
  - `nDCG@10 = 0.8570`
  - `MRR = 0.8434`
- `Evidence Recall@10 = 0.9786`.
- `Evidence Recall@15 = 1.0000`.
- `OOD AUROC = 1.0000` trên TEST benchmark.
- Query embedding sử dụng instruction thông qua `prompt_name="query"`.

---

## Dense Candidate Pool

**Selected: `top_k = 15`**

Ở TEST:

```text
Recall@10 = 0.9786
Recall@15 = 1.0000
```

Ba query chưa được phủ đầy đủ ở Top-10 đều được retrieve trong Top-15.

Điều này cho phép giữ candidate pool đủ rộng trước các bước hybrid retrieval và post-processing.

---

# Lưu ý về kết quả benchmark

Các metric ở README này thuộc **chunking + dense embedding retrieval benchmark**.

Chúng không nên được diễn giải là accuracy của chatbot hoàn chỉnh.

Để đánh giá end-to-end RAG cần tiếp tục đo riêng các thành phần như:

```text
Raw Query
   ↓
Query Rewriting
   ↓
Dense + BM25
   ↓
RRF
   ↓
Post-processing
   ↓
Generation
   ↓
Citation / Faithfulness / Answer Correctness
```

Các experiment tiếp theo nên giữ nguyên cấu hình chunking và embedding đã chốt ở benchmark này để tránh tuning lại dựa trên TEST.

---

# Tech Stack

- Python
- Sentence Transformers
- Qwen3 Embedding
- Qdrant
- BM25
- underthesea
- Ollama
- Qwen3
- Docling

---

# Repository Notes

Repository không lưu các artifact dữ liệu và indexing cục bộ.

Nên giữ ngoài Git các thành phần như:

```text
raw/intermediate documents
cleaned documents
vector database
BM25 serialized index
large model files
temporary evaluation artifacts
```

Chỉ source code, configuration, evaluation code và các file cần thiết để tái tạo pipeline nên được version control.
