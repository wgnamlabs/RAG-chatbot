"""
run_rerank_eval.py — So sánh Hybrid vs Hybrid+Rerank, thêm vào retrieval_comparison.csv.

Bảng "progressive improvement" đầy đủ:

  Method                              | Recall@5 | Recall@10 | Prec@3 | MRR
  ----------------------------------- | -------- | --------- | ------ | ----
  Dense only                          | ...      | ...       | ...    | ...
  BM25 only                           | ...      | ...       | ...    | ...
  Hybrid RRF (Dense + BM25)           | ...      | ...       | ...    | ...
  Hybrid + Rerank@5 (bge-reranker-v2) | ...      | N/A       | ...    | ...
  Hybrid + Rerank@3 (bge-reranker-v2) | N/A      | N/A       | ...    | ...

Chạy:
    python evaluation/run_rerank_eval.py
    python evaluation/run_rerank_eval.py --device cpu
    python evaluation/run_rerank_eval.py --candidate_k 30  # lấy 30 từ hybrid trước khi rerank
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
from generation.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from metrics import recall_at_k, precision_at_k, mean_reciprocal_rank


def evaluate_results_list(results_per_question: List[List], ground_truths, in_domain_idx,
                           k_values=(3, 5, 10)):
    """Tính metrics từ danh sách kết quả đã retrieve."""
    metrics_sum = {k: {"recall": 0.0, "precision": 0.0} for k in k_values}
    mrr_sum = 0.0
    n_in = len(in_domain_idx)

    for i, q_idx in enumerate(in_domain_idx):
        retrieved_meta = [r.metadata for r in results_per_question[i]]
        for k in k_values:
            metrics_sum[k]["recall"]    += recall_at_k(retrieved_meta, ground_truths[q_idx], k)
            metrics_sum[k]["precision"] += precision_at_k(retrieved_meta, ground_truths[q_idx], k)
        mrr_sum += mean_reciprocal_rank(retrieved_meta, ground_truths[q_idx])

    out = {}
    for k in k_values:
        out[f"recall@{k}"]    = round(metrics_sum[k]["recall"]    / n_in, 4)
        out[f"precision@{k}"] = round(metrics_sum[k]["precision"] / n_in, 4)
    out["mrr"] = round(mrr_sum / n_in, 4)
    return out


def print_progressive_table(rows: list) -> None:
    sep = "=" * 75
    print(f"\n{sep}")
    print(f"{'Method':<40} | {'Recall@5':>8} | {'Recall@10':>9} | {'Prec@3':>6} | {'MRR':>6}")
    print(f"{'-'*75}")
    for r in rows:
        rec5  = str(r.get("recall@5",  "N/A"))
        rec10 = str(r.get("recall@10", "N/A"))
        p3    = str(r.get("precision@3", "N/A"))
        mrr   = str(r.get("mrr", "N/A"))
        print(f"{r['method']:<40} | {rec5:>8} | {rec10:>9} | {p3:>6} | {mrr:>6}")
    print(f"{sep}\n")


def run_rerank_eval(base_path: Path, device: str = "cuda", candidate_k: int = 50) -> None:
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

    # ── Load retriever stack ────────────────────────────────────────────────────
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

    # Load reranker
    reranker_config = RerankerConfig(model_name="BAAI/bge-reranker-v2-m3", device="auto")
    reranker = CrossEncoderReranker(reranker_config)
    reranker.load()

    # ── Collect results ─────────────────────────────────────────────────────────
    print(f"\n[Rerank Eval] Đang retrieve {candidate_k} candidates mỗi câu và rerank...")
    hybrid_results_all  = []  # top-10 hybrid
    reranked5_all       = []  # rerank → top-5
    reranked3_all       = []  # rerank → top-3

    for q_idx in in_domain_idx:
        q = questions[q_idx]

        # Hybrid lấy nhiều candidates hơn để reranker có đủ input
        candidates = hybrid_retriever.retrieve(q, top_k=candidate_k)

        hybrid_results_all.append(candidates[:10])  # top-10 hybrid (không rerank)
        reranked5_all.append(reranker.rerank(q, candidates, top_k=5))
        reranked3_all.append(reranker.rerank(q, candidates, top_k=3))

    reranker.unload()

    # ── Metrics ─────────────────────────────────────────────────────────────────
    hybrid_metrics   = evaluate_results_list(hybrid_results_all, ground_truths, in_domain_idx)
    reranked5_metrics = evaluate_results_list(reranked5_all,     ground_truths, in_domain_idx)
    reranked3_metrics = evaluate_results_list(reranked3_all,     ground_truths, in_domain_idx)

    # ── Load existing rows từ retrieval_comparison.csv ──────────────────────────
    existing_rows = []
    fieldnames = ["method", "split", "k", "recall@k", "precision@k", "mrr", "top1_sim_ood"]
    if output_csv.exists():
        with open(output_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = [r for r in reader
                             if "Rerank" not in r.get("method", "")]

    # Build new rows to append
    def build_csv_rows(method_name, metrics_dict, k_values=(3, 5, 10)):
        rows = []
        for k in k_values:
            rows.append({
                "method":      method_name,
                "split":       "in_domain",
                "k":           k,
                "recall@k":    metrics_dict.get(f"recall@{k}", ""),
                "precision@k": metrics_dict.get(f"precision@{k}", ""),
                "mrr":         metrics_dict["mrr"],
                "top1_sim_ood": "",
            })
        return rows

    new_rows = []
    new_rows += build_csv_rows("Hybrid RRF (Dense+BM25) top10",    hybrid_metrics)
    new_rows += build_csv_rows("Hybrid + bge-reranker-v2-m3 top5", reranked5_metrics, (3, 5))
    new_rows += build_csv_rows("Hybrid + bge-reranker-v2-m3 top3", reranked3_metrics, (3,))

    all_rows = existing_rows + new_rows
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # ── Print progressive table ─────────────────────────────────────────────────
    table_rows = [
        {"method": "Hybrid RRF (Dense+BM25) top10",    **hybrid_metrics},
        {"method": "Hybrid + bge-reranker-v2-m3 top5", **reranked5_metrics,
         "recall@10": "N/A"},
        {"method": "Hybrid + bge-reranker-v2-m3 top3", **reranked3_metrics,
         "recall@5": "N/A", "recall@10": "N/A"},
    ]
    print_progressive_table(table_rows)
    print(f"✅ Kết quả lưu tại {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",      default="cuda", help="cuda | cpu")
    parser.add_argument("--candidate_k", default=50,  type=int,
                        help="Số candidates hybrid lấy trước khi rerank (khuyến nghị 50)")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent
    run_rerank_eval(base_path, device=args.device, candidate_k=args.candidate_k)
