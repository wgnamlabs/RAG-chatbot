# RAG Chatbot Tư Vấn Phụ Sản

Hệ thống **Retrieval-Augmented Generation (RAG)** hỗ trợ tra cứu và tư vấn thông tin sức khỏe sản phụ khoa bằng tiếng Việt.

Dự án được phát triển theo hướng **local-first**, ưu tiên khả năng kiểm soát pipeline, đánh giá độc lập từng thành phần và đảm bảo các quyết định kỹ thuật được lựa chọn trên tập phát triển trước khi xác nhận trên tập kiểm thử.

> **Trạng thái hiện tại:** Offline indexing pipeline đã hoàn thiện. Chiến lược chunking và embedding đã được lựa chọn, VectorDB đã được xây dựng thành công trên Qdrant và sparse index BM25 đã sẵn sàng. Phần retrieval và generation online chưa nằm trong milestone hiện tại.

---

## Mục tiêu dự án

Dự án hướng tới xây dựng một chatbot RAG tiếng Việt cho lĩnh vực sản phụ khoa với các mục tiêu chính:

- truy xuất chính xác các đoạn thông tin liên quan từ kho tri thức chuyên ngành;
- bảo toàn cấu trúc tài liệu trong quá trình chia chunk;
- hỗ trợ tốt các câu hỏi cần nhiều bằng chứng, số liệu và nội dung dạng bảng;
- đánh giá riêng từng thành phần trước khi tích hợp thành pipeline RAG hoàn chỉnh;
- triển khai được trên môi trường local với kiến trúc dễ kiểm soát và tái lập;
- hạn chế phụ thuộc vào dịch vụ bên ngoài đối với các thành phần cốt lõi.

---

## Trạng thái phát triển

Pipeline hiện tại đã hoàn thành:

```text
Document Processing
        ↓
Text Cleaning
        ↓
Chunking Evaluation
        ↓
Hierarchical Chunking
        ↓
Embedding Evaluation
        ↓
Qwen3-Embedding-4B
        ↓
Vector Store Construction
        ↓
Qdrant + BM25 Index
```

Các thành phần sau **chưa nằm trong phạm vi hiện tại**:

```text
Query Rewriting
Hybrid Retrieval
RRF Fusion
Reranking
Prompt Construction
Answer Generation
Citation / Faithfulness Validation
Safety Layer
Web UI
```

---

# Kiến trúc hiện tại

## Offline Indexing Pipeline

```text
Source Documents
       │
       ▼
Document Loader
       │
       ▼
Text Cleaning
       │
       ▼
Hierarchical Chunking
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
Dense Embedding          Sparse Index
Qwen3-Embedding-4B          BM25
       │                     │
       ▼                     ▼
Qdrant Vector Store      BM25 Index
```

Pipeline indexing được thiết kế theo hướng module hóa để từng bước có thể được benchmark, kiểm thử và thay thế độc lập.

---

## Chunking

Dự án triển khai nhiều chiến lược chunking để benchmark trước khi lựa chọn cấu hình chính thức.

Hai hướng chính đã được đánh giá:

- **Semantic Chunking**
- **Hierarchical Chunking**

Cấu hình được lựa chọn:

```text
Chunking Strategy: Hierarchical Chunking
```

Hierarchical Chunking được ưu tiên vì:

- bảo toàn tốt cấu trúc heading của tài liệu;
- duy trì breadcrumb/section context trong từng chunk;
- hoạt động ổn định với nội dung nhiều cấp heading;
- hỗ trợ tốt nội dung dạng bảng và multi-evidence;
- cho chất lượng ranking tốt khi kết hợp với embedding model được chọn.

Implementation:

```text
src/indexing/chunking/
├── base.py
├── config.py
├── hierarchical_chunker.py
├── markdown_utils.py
├── semantic_chunker.py
└── table_utils.py
```

---

## Embedding

Embedding layer được thiết kế theo interface riêng để có thể benchmark và thay đổi model mà không ảnh hưởng tới các tầng khác.

Các model đã được đánh giá:

- `AITeamVN/Vietnamese_Embedding_v2`
- `Qwen/Qwen3-Embedding-4B`
- `BAAI/bge-m3`

Model được lựa chọn chính thức:

```text
Qwen/Qwen3-Embedding-4B
```

Cấu hình:

```text
Vector dimension : 2560
Similarity       : Cosine
```

Khi encode:

- document/chunk được encode theo chế độ document embedding;
- query embedding sử dụng query instruction tương ứng với model.

Implementation:

```text
src/indexing/embedding/
├── base.py
├── config.py
└── embedder.py
```

---

## Vector Store

Vector store sử dụng **Qdrant** làm dense vector database.

Cấu hình hiện tại:

```text
Collection       : phu_san_chunks
Vector dimension : 2560
Distance         : Cosine
```

Sparse index được xây dựng song song bằng **BM25**.

Artifact chính sau indexing:

```text
data/vector_db/
├── qdrant/
├── bm25_index.pkl
└── index_manifest.json
```

`index_manifest.json` lưu metadata của quá trình build index, giúp kiểm tra tính nhất quán giữa chunking, embedding và vector store.

Implementation:

```text
src/indexing/vector_store/
├── base.py
├── config.py
└── qdrant_store.py
```

---

# Evaluation Strategy

Dự án áp dụng quy trình đánh giá theo hai split:

```text
DEV  → lựa chọn cấu hình
TEST → xác nhận khả năng tổng quát hóa
```

Nguyên tắc:

1. các lựa chọn về chunking và embedding được thực hiện trên DEV;
2. TEST chỉ được sử dụng sau khi cấu hình đã được freeze;
3. không thay đổi model dựa trên kết quả TEST;
4. ground truth được đánh giá ở mức **gold evidence**;
5. evidence được map độc lập vào output của từng chunker trước khi tính metric.

Các metric chính:

| Metric | Ý nghĩa |
|---|---|
| **Evidence Recall@K** | Tỷ lệ gold evidence được retrieve trong Top-K |
| **Evidence Complete@K** | Tỷ lệ câu hỏi retrieve đủ toàn bộ evidence cần thiết |
| **MRR** | Vị trí của relevant result đầu tiên |
| **nDCG@K** | Chất lượng thứ hạng của relevant chunks |
| **OOD AUROC** | Khả năng phân biệt câu hỏi in-domain và out-of-domain |

Ưu tiên khi lựa chọn cấu hình:

```text
Evidence Recall / Evidence Completeness
                ↓
          Ranking Quality
            MRR / nDCG
```

---

# Benchmark Results

## Selected Configuration on DEV

Cấu hình được freeze trước khi mở TEST:

```text
Chunking  : Hierarchical
Embedding : Qwen/Qwen3-Embedding-4B
```

| Metric | DEV |
|---|---:|
| Evidence Recall@10 | **1.0000** |
| Evidence Complete@10 | **1.0000** |
| nDCG@10 | **0.8961** |
| MRR | **0.8773** |
| OOD AUROC | **1.0000** |

---

## Final Dense Retrieval TEST

| Metric | TEST |
|---|---:|
| Evidence Recall@10 | **0.9786** |
| Evidence Complete@10 | **0.9786** |
| nDCG@10 | **0.8570** |
| MRR | **0.8434** |
| OOD AUROC | **1.0000** |

Hiệu năng giảm nhẹ từ DEV sang TEST nhưng vẫn duy trì evidence coverage và ranking quality cao, cho thấy cấu hình đã chọn tổng quát hóa tốt trên tập giữ lại.

---

## Recall theo K

Với `Hierarchical + Qwen3-Embedding-4B`:

| K | Evidence Recall |
|---:|---:|
| 3 | 0.9000 |
| 5 | 0.9464 |
| 10 | 0.9786 |
| 15 | **1.0000** |

Top-15 được xem là candidate pool hợp lý cho giai đoạn retrieval online trong tương lai.

---

## Hard / Multi-evidence Queries

| K | Evidence Recall | Evidence Complete |
|---:|---:|---:|
| 3 | 0.7083 | 0.5417 |
| 5 | 0.8542 | 0.7500 |
| 10 | 0.9583 | 0.9583 |
| 15 | **1.0000** | **1.0000** |

Các câu hỏi cần nhiều evidence khó hơn rõ rệt ở K nhỏ. Kết quả này là cơ sở để giữ candidate pool đủ rộng trước các bước retrieval/context selection ở giai đoạn sau.

---

## Table Queries

| K | Evidence Recall |
|---:|---:|
| 3 | 0.9500 |
| 5 | **1.0000** |
| 10 | **1.0000** |

Hierarchical Chunking duy trì hiệu quả tốt với nội dung có cấu trúc bảng.

---

# Project Structure

```text
rag-phu-san-chatbot/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── cleaned/
│   └── vector_db/
│       ├── qdrant/
│       ├── bm25_index.pkl
│       └── index_manifest.json
│
├── evaluation/
│   ├── results/
│   │   ├── chunks_cache/
│   │   └── embedding_eval/
│   ├── build_vector_store.py
│   ├── eval_questions.json
│   ├── eval_questions_manifest.json
│   ├── EVAL_SCHEMA.md
│   ├── metrics.py
│   ├── run_chunking_eval.py
│   └── run_embedding_eval.py
│
├── src/
│   ├── __init__.py
│   └── indexing/
│       ├── __init__.py
│       ├── loader.py
│       ├── cleaner.py
│       ├── chunking/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── config.py
│       │   ├── hierarchical_chunker.py
│       │   ├── markdown_utils.py
│       │   ├── semantic_chunker.py
│       │   └── table_utils.py
│       ├── embedding/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── config.py
│       │   └── embedder.py
│       └── vector_store/
│           ├── __init__.py
│           ├── base.py
│           ├── config.py
│           └── qdrant_store.py
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# Core Modules

### `src/indexing/loader.py`

Đọc tài liệu đầu vào và chuyển nội dung sang representation phù hợp cho preprocessing.

### `src/indexing/cleaner.py`

Chuẩn hóa nội dung trước chunking trong khi vẫn bảo toàn cấu trúc semantic cần thiết.

### `src/indexing/chunking/`

Chứa abstraction và implementation cho các chiến lược chunking.

### `src/indexing/embedding/`

Đóng gói embedding model và interface encode document/query.

### `src/indexing/vector_store/`

Quản lý Qdrant collection, vector insertion và các primitive cần thiết cho retrieval ở giai đoạn sau.

### `evaluation/`

Chứa benchmark protocol, metrics và các script phục vụ lựa chọn chunking/embedding.

---

# Installation

## 1. Clone repository

```bash
git clone <repository-url>
cd rag-phu-san-chatbot
```

## 2. Create virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Hoặc:

```bash
pip install -e .
```

---

# Environment Configuration

Tạo `.env` từ template.

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

### Linux / macOS

```bash
cp .env.example .env
```

Không commit `.env` lên repository.

---

# Running the Pipeline

## Chunking Evaluation

DEV:

```bash
python evaluation/run_chunking_eval.py --split dev
```

TEST:

```bash
python evaluation/run_chunking_eval.py --split test
```

TEST chỉ được chạy sau khi cấu hình đã được freeze trên DEV.

---

## Embedding Evaluation

DEV:

```bash
python evaluation/run_embedding_eval.py --split dev
```

TEST:

```bash
python evaluation/run_embedding_eval.py --split test
```

---

## Build Vector Store

Sau khi chunking và embedding đã được lựa chọn:

```bash
python evaluation/build_vector_store.py
```

Script chịu trách nhiệm tạo các artifact indexing cần thiết cho Qdrant và BM25.

---

# Current Technical Decisions

| Component | Selected Configuration |
|---|---|
| Chunking | Hierarchical Chunking |
| Embedding | `Qwen/Qwen3-Embedding-4B` |
| Vector dimension | 2560 |
| Similarity | Cosine |
| Vector Database | Qdrant |
| Collection | `phu_san_chunks` |
| Sparse Index | BM25 |
| Candidate pool cho retrieval phase tiếp theo | Top-15 |

Các quyết định trên được xem là **frozen** tại milestone hiện tại.

---

# Tech Stack

- **Python**
- **Docling**
- **Sentence Transformers**
- **Qwen3 Embedding**
- **Qdrant / qdrant-client**
- **BM25 / rank-bm25**
- **pandas / NumPy**

---

# Repository Policy

Repository chỉ version-control source code, configuration và evaluation logic cần thiết để tái tạo pipeline.

Các artifact local nên được loại khỏi Git qua `.gitignore`, bao gồm:

```text
virtual environments
raw/intermediate artifacts
local vector database
serialized BM25 index
model cache
temporary evaluation outputs
Python cache files
```

Không commit:

```text
.env
.venv/
__pycache__/
```

---

# Development Roadmap

```text
[✓] Document processing
[✓] Text cleaning
[✓] Chunking benchmark
[✓] Embedding benchmark
[✓] Hierarchical chunking selected
[✓] Qwen3-Embedding-4B selected
[✓] Qdrant vector store built
[✓] BM25 index built

[ ] Dense retrieval module
[ ] Sparse retrieval module
[ ] Hybrid retrieval
[ ] Query rewriting
[ ] Retrieval evaluation
[ ] Prompt construction
[ ] Answer generation
[ ] Citation / faithfulness evaluation
[ ] Safety layer
[ ] Application interface
```

---

# Notes

Các benchmark trong README hiện tại chỉ phản ánh chất lượng của **offline indexing và dense retrieval evaluation**.

Chúng **không phải accuracy của chatbot hoàn chỉnh**.

Milestone hiện tại đã hoàn thiện và freeze:

```text
Chunking
Embedding
Vector Store
```

Giai đoạn tiếp theo sẽ tập trung vào retrieval online dựa trên VectorDB đã được xây dựng.
