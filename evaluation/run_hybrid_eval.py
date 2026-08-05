"""
run_hybrid_eval.py — So sánh Dense-only vs BM25-only vs Hybrid (nhiều cấu hình RRF).

Cấu hình RRF được test:
  1. Dense only
  2. BM25 only
  3. Hybrid RRF 1:1  (dense_weight=1.0, bm25_weight=1.0)
  4. Hybrid RRF 1:2  (dense_weight=1.0, bm25_weight=2.0) — BM25 heavy
  5. Hybrid RRF 1:3  (dense_weight=1.0, bm25_weight=3.0) — BM25 dominant

Output: evaluation/results/retrieval_comparison.csv

Chạy:
    python evaluation/run_hybrid_eval.py
    python evaluation/run_hybrid_eval.py --device cpu
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

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
from metrics import recall_at_k, precision_at_k, mean_reciprocal_rank


# ── Helpers ────────────────────────────────────────────────────────────────────

def evaluate_retriever(retriever, questions, ground_truths, in_domain_idx, out_domain_idx,
                       k_values=(3, 5, 10)):
    """Chạy retriever trên toàn bộ test set, trả về dict kết quả."""
    metrics_sum = {k: {"recall": 0.0, "precision": 0.0} for k in k_values}
    mrr_sum = 0.0
    n_in = len(in_domain_idx)

    for q_idx in in_domain_idx:
        results = retriever.retrieve(questions[q_idx], top_k=max(k_values))
        retrieved_meta = [r.metadata for r in results]

        for k in k_values:
            metrics_sum[k]["recall"]    += recall_at_k(retrieved_meta, ground_truths[q_idx], k)
            metrics_sum[k]["precision"] += precision_at_k(retrieved_meta, ground_truths[q_idx], k)
        mrr_sum += mean_reciprocal_rank(retrieved_meta, ground_truths[q_idx])

    result_dict = {}
    for k in k_values:
        result_dict[f"recall@{k}"]    = round(metrics_sum[k]["recall"]    / n_in, 4)
        result_dict[f"precision@{k}"] = round(metrics_sum[k]["precision"] / n_in, 4)
    result_dict["mrr"] = round(mrr_sum / n_in, 4)

    # OOD: top-1 score
    if out_domain_idx:
        top1_scores = []
        for q_idx in out_domain_idx:
            res = retriever.retrieve(questions[q_idx], top_k=1)
            if res:
                top1_scores.append(res[0].score)
        result_dict["top1_sim_ood"] = round(sum(top1_scores) / len(top1_scores), 4) if top1_scores else 0.0
    else:
        result_dict["top1_sim_ood"] = ""

    return result_dict


def print_comparison_table(rows: list) -> None:
    """In bảng so sánh đầy đủ ra console."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"{'Method':<40} | {'R@5':>6} | {'R@10':>6} | {'P@3':>6} | {'MRR':>6}")
    print(f"{'-'*72}")
    best_mrr = max(r.get("mrr", 0) for r in rows)
    for row in rows:
        mrr   = row.get("mrr", "")
        star  = " ★" if mrr == best_mrr else "  "
        r5    = str(row.get("recall@5",  "-"))
        r10   = str(row.get("recall@10", "-"))
        p3    = str(row.get("precision@3", "-"))
        print(f"{row['method']:<40} | {r5:>6} | {r10:>6} | {p3:>6} | {mrr:>6}{star}")
    print(f"{sep}\n")
    print(f"  ★ = MRR cao nhất")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_hybrid_eval(base_path: Path, device: str = "cuda") -> None:
    output_csv = base_path / "evaluation" / "results" / "retrieval_comparison.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── Load Questions ──────────────────────────────────────────────────────────
    questions_path = base_path / "evaluation" / "eval_questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    questions     = [item["question"]             for item in qa_data]
    ground_truths = [item["ground_truth_sources"] for item in qa_data]
    in_domain_idx  = [i for i, gt in enumerate(ground_truths) if gt]
    out_domain_idx = [i for i, gt in enumerate(ground_truths) if not gt]
    print(f"[Hybrid Eval] {len(questions)} câu hỏi "
          f"(in-domain: {len(in_domain_idx)}, OOD: {len(out_domain_idx)})")

    # ── Load BM25 ───────────────────────────────────────────────────────────────
    bm25_pickle = base_path / "data" / "vector_db" / "bm25_index.pkl"
    if not bm25_pickle.exists():
        print(f"❌ BM25 index không tồn tại: {bm25_pickle}")
        print("   Hãy chạy: python evaluation/build_vector_store.py")
        sys.exit(1)
    bm25_retriever = BM25Retriever()
    bm25_retriever.load_from_pickle(bm25_pickle)

    # ── Load Dense ─────────────────────────────────────────────────────────────
    emb_config = EmbedderConfig(
        model_name="AITeamVN/Vietnamese_Embedding",
        batch_size=32, max_seq_length=4096, device=device,
    )
    embedder = SentenceTransformerEmbedder(emb_config)
    embedder.load()

    qdrant_config = QdrantStoreConfig(host="localhost", port=6333)
    store = QdrantVectorStore(config=qdrant_config)
    store.load()

    dense_retriever = DenseRetriever(store=store, embedder=embedder)

    # ── Tạo tất cả cấu hình Hybrid ─────────────────────────────────────────────
    hybrid_configs = [
        ("Hybrid RRF 1:1  (dense=1, bm25=1)", HybridRetrieverConfig(rrf_k=60, dense_weight=1.0, bm25_weight=1.0)),
        ("Hybrid RRF 1:2  (dense=1, bm25=2)", HybridRetrieverConfig(rrf_k=60, dense_weight=1.0, bm25_weight=2.0)),
        ("Hybrid RRF 1:3  (dense=1, bm25=3)", HybridRetrieverConfig(rrf_k=60, dense_weight=1.0, bm25_weight=3.0)),
    ]

    # ── Evaluate tất cả ────────────────────────────────────────────────────────
    k_vals = (3, 5, 10)
    all_results = []
    csv_rows = []

    # Dense only
    print(f"\n[Hybrid Eval] Dense only...")
    m = evaluate_retriever(dense_retriever, questions, ground_truths, in_domain_idx, out_domain_idx, k_vals)
    m["method"] = "Dense only (Vietnamese_Embedding+semantic)"
    all_results.append(m)

    # BM25 only
    print(f"[Hybrid Eval] BM25 only...")
    m = evaluate_retriever(bm25_retriever, questions, ground_truths, in_domain_idx, out_domain_idx, k_vals)
    m["method"] = "BM25 only (underthesea)"
    all_results.append(m)

    # Hybrid variants
    for name, cfg in hybrid_configs:
        print(f"[Hybrid Eval] {name}...")
        hybrid = HybridRetriever(dense_retriever, bm25_retriever, cfg)
        m = evaluate_retriever(hybrid, questions, ground_truths, in_domain_idx, out_domain_idx, k_vals)
        m["method"] = name
        all_results.append(m)

    # ── Print table ─────────────────────────────────────────────────────────────
    print_comparison_table(all_results)

    # ── Best config recommendation ─────────────────────────────────────────────
    best = max(all_results, key=lambda r: r.get("mrr", 0))
    print(f"💡 Config tốt nhất theo MRR: [{best['method']}]")
    print(f"   → Dùng config này làm default trong pipeline\n")

    # ── Write CSV (ghi đè toàn bộ phần hybrid, giữ rerank rows nếu có) ─────────
    fieldnames = ["method", "split", "k", "recall@k", "precision@k", "mrr", "top1_sim_ood"]

    # Load rows cũ chỉ giữ các dòng Rerank (nếu đã chạy run_rerank_eval trước)
    existing_rerank_rows = []
    if output_csv.exists():
        with open(output_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_rerank_rows = [r for r in reader if "Rerank" in r.get("method", "")]

    new_rows = []
    for result in all_results:
        for k in k_vals:
            new_rows.append({
                "method":      result["method"],
                "split":       "in_domain",
                "k":           k,
                "recall@k":    result.get(f"recall@{k}", ""),
                "precision@k": result.get(f"precision@{k}", ""),
                "mrr":         result["mrr"],
                "top1_sim_ood": "",
            })
        new_rows.append({
            "method":      result["method"],
            "split":       "out_of_domain",
            "k":           "",
            "recall@k":    "",
            "precision@k": "",
            "mrr":         "",
            "top1_sim_ood": result.get("top1_sim_ood", ""),
        })

    all_rows = new_rows + existing_rerank_rows
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"✅ Kết quả lưu tại {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="cuda | cpu")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent
    run_hybrid_eval(base_path, device=args.device)
