"""
error_analysis.py — Phân tích các câu hỏi bị miss sau Hybrid+Rerank.

Phân loại lỗi heuristic:
  A. ABBREVIATION/SYNONYM  — Query dùng viết tắt y khoa hoặc từ đồng nghĩa
                              không khớp với chunk (ĐTĐ vs đái tháo đường, v.v.)
  B. CHUNKING BOUNDARY     — Ground truth nằm ở biên chunk, bị cắt sai
                             (dấu hiệu: top result cùng source nhưng chunk khác)
  C. MULTI-HOP             — Query yêu cầu kết hợp thông tin từ nhiều đoạn

Output: evaluation/results/error_analysis_report.md

Chạy:
    python evaluation/error_analysis.py
    python evaluation/error_analysis.py --device cpu --candidate_k 20 --final_k 10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Reconfigure stdout for Windows console
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indexing.embedding import SentenceTransformerEmbedder
from indexing.embedding.config import EmbedderConfig
from indexing.vector_store import QdrantVectorStore, QdrantStoreConfig
from generation.retrieval.bm25_retriever import BM25Retriever
from generation.retrieval.dense_retriever import DenseRetriever
from generation.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig
from generation.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from generation.retrieval.base import RetrievalResult
from metrics import get_doc_id, recall_at_k


# ── Medical abbreviations / synonyms ──────────────────────────────────────────
MEDICAL_ABBREVS = {
    "ĐTĐ":    ["đái tháo đường", "tiểu đường"],
    "GDM":    ["gestational diabetes", "đái tháo đường thai kỳ"],
    "TSG":    ["tiền sản giật", "pre-eclampsia"],
    "SG":     ["sản giật", "eclampsia"],
    "MLT":    ["mổ lấy thai", "mổ caesar"],
    "DCTC":   ["dụng cụ tử cung", "vòng tránh thai"],
    "KHHGĐ":  ["kế hoạch hóa gia đình"],
    "SKSS":   ["sức khỏe sinh sản"],
}

def _has_abbreviation(query: str) -> bool:
    query_lower = query.lower()
    for abbrev in MEDICAL_ABBREVS:
        if abbrev.lower() in query_lower:
            return True
    # Kiểm tra thêm: query có từ ngắn viết hoa (pattern của viết tắt y khoa)
    import re
    return bool(re.search(r'\b[A-ZĐÀÁẠẢÃÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸ]{2,5}\b', query))


def _classify_error(
    query: str,
    ground_truth_sources: List[str],
    top10_results: List[RetrievalResult],
) -> str:
    """Phân loại lỗi heuristic."""
    top10_meta = [r.metadata for r in top10_results]
    top10_sources = [get_doc_id(m) for m in top10_meta]

    # Nếu query chứa viết tắt → có thể lỗi synonym
    if _has_abbreviation(query):
        return "A. ABBREVIATION/SYNONYM"

    # Nếu top-10 có cùng source file nhưng chunk khác → chunking boundary
    gt_set = set(ground_truth_sources)
    same_source_count = sum(1 for s in top10_sources if s in gt_set)
    if same_source_count > 0:
        return "B. CHUNKING BOUNDARY"

    # Query có từ ghép phức tạp hoặc câu hỏi có nhiều điểm hỏi → multi-hop
    multi_hop_signals = ["và", "cũng như", "đồng thời", "kết hợp", "ngoài ra", "so sánh"]
    if any(signal in query.lower() for signal in multi_hop_signals):
        return "C. MULTI-HOP"

    return "B. CHUNKING BOUNDARY"  # default


# ── Main ───────────────────────────────────────────────────────────────────────

def run_error_analysis(
    base_path: Path,
    device: str = "cuda",
    candidate_k: int = 20,
    final_k: int = 10,
) -> None:
    report_path = base_path / "evaluation" / "results" / "error_analysis_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Load questions
    questions_path = base_path / "evaluation" / "eval_questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    questions     = [item["question"]             for item in qa_data]
    ground_truths = [item["ground_truth_sources"] for item in qa_data]
    in_domain_idx  = [i for i, gt in enumerate(ground_truths) if gt]

    # Load retriever stack
    bm25_pickle = base_path / "data" / "vector_db" / "bm25_index.pkl"
    if not bm25_pickle.exists():
        print(f"❌ BM25 index không tồn tại. Chạy build_vector_store.py trước.")
        sys.exit(1)

    bm25_retriever = BM25Retriever()
    bm25_retriever.load_from_pickle(bm25_pickle)

    emb_config = EmbedderConfig(
        model_name="AITeamVN/Vietnamese_Embedding",
        batch_size=32, max_seq_length=4096, device=device,
    )
    embedder = SentenceTransformerEmbedder(emb_config)
    embedder.load()

    qdrant_config = QdrantStoreConfig(
        host="localhost", port=6333,
    )
    store = QdrantVectorStore(config=qdrant_config)
    store.load()

    dense_retriever  = DenseRetriever(store=store, embedder=embedder)
    hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever,
                                       HybridRetrieverConfig(rrf_k=60))

    reranker = CrossEncoderReranker(RerankerConfig(model_name="BAAI/bge-reranker-v2-m3"))
    reranker.load()

    # ── Collect miss cases ──────────────────────────────────────────────────────
    miss_cases = []
    hit_count  = 0

    print(f"\n[Error Analysis] Chạy pipeline trên {len(in_domain_idx)} câu hỏi in-domain...")
    for q_idx in in_domain_idx:
        q   = questions[q_idx]
        gts = ground_truths[q_idx]

        candidates = hybrid_retriever.retrieve(q, top_k=candidate_k)
        reranked   = reranker.rerank(q, candidates, top_k=final_k)

        recall = recall_at_k([r.metadata for r in reranked], gts, final_k)

        if recall == 0.0:
            error_type = _classify_error(q, gts, reranked)
            miss_cases.append({
                "question":     q,
                "ground_truth": gts,
                "top_results":  reranked,
                "error_type":   error_type,
            })
            print(f"  ❌ MISS [{error_type}]: {q[:60]}...")
        else:
            hit_count += 1
            print(f"  ✅ HIT : {q[:60]}...")

    reranker.unload()

    total = len(in_domain_idx)
    print(f"\n[Error Analysis] Hit: {hit_count}/{total}, Miss: {len(miss_cases)}/{total}")

    # ── Classify summary ────────────────────────────────────────────────────────
    from collections import Counter
    error_counts = Counter(c["error_type"] for c in miss_cases)

    # ── Write markdown report ───────────────────────────────────────────────────
    lines = [
        "# Error Analysis Report — RAG Phụ Sản Chatbot",
        "",
        f"**Pipeline**: Hybrid RRF (Dense + BM25) → bge-reranker-v2-m3 → top-{final_k}",
        f"**Ngày chạy**: {__import__('datetime').date.today()}",
        "",
        "## Tóm tắt",
        "",
        f"| Metric | Giá trị |",
        f"|--------|---------|",
        f"| Tổng câu hỏi in-domain | {total} |",
        f"| Hit (Recall@{final_k} > 0) | {hit_count} ({hit_count/total*100:.1f}%) |",
        f"| Miss | {len(miss_cases)} ({len(miss_cases)/total*100:.1f}%) |",
        "",
        "## Phân loại lỗi",
        "",
        "| Loại lỗi | Số lượng |",
        "|----------|----------|",
    ]
    for err_type, count in sorted(error_counts.items()):
        lines.append(f"| {err_type} | {count} |")

    lines += [
        "",
        "---",
        "",
        "## Chi tiết các câu hỏi bị miss",
        "",
    ]

    for i, case in enumerate(miss_cases, 1):
        lines += [
            f"### {i}. {case['question']}",
            "",
            f"**Phân loại lỗi**: `{case['error_type']}`",
            "",
            f"**Ground truth source**: {', '.join(case['ground_truth'])}",
            "",
            f"**Top-{final_k} kết quả thực tế trả về:**",
            "",
            "| Rank | Source | Score | Preview (100 ký tự đầu) |",
            "|------|--------|-------|--------------------------|",
        ]
        for r in case["top_results"]:
            src   = get_doc_id(r.metadata)
            score = f"{r.score:.4f}"
            preview = r.text[:100].replace("|", "｜").replace("\n", " ")
            lines.append(f"| {r.rank+1} | {src} | {score} | {preview}... |")

        lines += ["", "---", ""]

    lines += [
        "## Khuyến nghị cải thiện",
        "",
        "### A. ABBREVIATION/SYNONYM errors",
        "- Thêm bước query expansion: map viết tắt y khoa → từ đầy đủ trong `query_rewriter.py`.",
        "- Bổ sung synonym dictionary: ĐTĐ → 'đái tháo đường', GDM → 'gestational diabetes mellitus'.",
        "",
        "### B. CHUNKING BOUNDARY errors",
        "- Tăng `child_chunk_overlap` trong `HierarchicalChunkerConfig` (thử 300–400 ký tự).",
        "- Xem xét parent-document retrieval: lưu chunk con nhưng retrieve chunk cha khi cần context rộng hơn.",
        "",
        "### C. MULTI-HOP errors",
        "- Áp dụng query decomposition: chia câu hỏi phức tạp thành nhiều sub-queries.",
        "- Tăng `candidate_k` cho reranker (thử 50 thay vì 20).",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ Báo cáo lưu tại: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--candidate_k", default=20, type=int)
    parser.add_argument("--final_k",     default=10, type=int)
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent
    run_error_analysis(base_path, device=args.device,
                       candidate_k=args.candidate_k, final_k=args.final_k)
