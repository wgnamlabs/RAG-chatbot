"""
run_bm25_eval.py — Benchmark BM25-only retrieval trên eval_questions.json.

Kết quả append vào evaluation/results/comparison_matrix.csv
(chunker=bm25, embedding_model=-, giữ cùng format với run_embedding_eval.py).

Chạy:
    python evaluation/run_bm25_eval.py
    python evaluation/run_bm25_eval.py --chunker hierarchical
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Reconfigure stdout for Windows console
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generation.retrieval.bm25_retriever import BM25Retriever
from metrics import recall_at_k, precision_at_k, mean_reciprocal_rank


def run_bm25_eval(
    base_path: Path,
    chunker_name: str = "semantic",
    output_csv: Path = None,
) -> None:
    if output_csv is None:
        output_csv = base_path / "evaluation" / "results" / "comparison_matrix.csv"

    # 1. Load chunks
    chunk_file = base_path / "evaluation" / "results" / "chunks_cache" / f"{chunker_name}.json"
    if not chunk_file.exists():
        print(f"❌ Không tìm thấy {chunk_file}. Chạy run_chunking_eval.py trước.")
        sys.exit(1)

    print(f"[BM25 Eval] Load chunks từ {chunk_file}...")
    with open(chunk_file, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)
    print(f"  → {len(raw_chunks)} chunks.")

    # 2. Build BM25
    retriever = BM25Retriever(chunks=raw_chunks)

    # 3. Load questions
    questions_path = base_path / "evaluation" / "eval_questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    questions     = [item["question"]              for item in qa_data]
    ground_truths = [item["ground_truth_sources"]  for item in qa_data]

    in_domain_idx  = [i for i, gt in enumerate(ground_truths) if gt]
    out_domain_idx = [i for i, gt in enumerate(ground_truths) if not gt]
    print(f"[BM25 Eval] Câu hỏi: {len(questions)} "
          f"(in-domain: {len(in_domain_idx)}, out-of-domain: {len(out_domain_idx)})")

    # 4. Evaluate in-domain
    k_values = [3, 5, 10]
    metrics_sum = {k: {"recall": 0.0, "precision": 0.0} for k in k_values}
    mrr_sum = 0.0

    for q_idx in in_domain_idx:
        results = retriever.retrieve(questions[q_idx], top_k=max(k_values))
        retrieved_metadata = [r.metadata for r in results]

        for k in k_values:
            metrics_sum[k]["recall"]    += recall_at_k(retrieved_metadata, ground_truths[q_idx], k)
            metrics_sum[k]["precision"] += precision_at_k(retrieved_metadata, ground_truths[q_idx], k)
        mrr_sum += mean_reciprocal_rank(retrieved_metadata, ground_truths[q_idx])

    n_in = len(in_domain_idx)
    new_rows = []

    print(f"\n{'='*60}")
    print(f"  BM25 × {chunker_name} (in-domain)")
    print(f"{'='*60}")
    for k in k_values:
        avg_recall    = metrics_sum[k]["recall"]    / n_in
        avg_precision = metrics_sum[k]["precision"] / n_in
        avg_mrr       = mrr_sum / n_in
        print(f"  k={k}: Recall={avg_recall:.4f}, Precision={avg_precision:.4f}, MRR={avg_mrr:.4f}")
        new_rows.append({
            "chunker":         "bm25",
            "embedding_model": "-",
            "split":           "in_domain",
            "k":               k,
            "recall@k":        f"{avg_recall:.4f}",
            "precision@k":     f"{avg_precision:.4f}",
            "mrr":             f"{avg_mrr:.4f}",
            "top1_sim_ood":    "",
        })

    # 5. Evaluate out-of-domain (top-1 BM25 score)
    if out_domain_idx:
        top1_scores = []
        for q_idx in out_domain_idx:
            results = retriever.retrieve(questions[q_idx], top_k=1)
            if results:
                top1_scores.append(results[0].score)

        avg_ood = sum(top1_scores) / len(top1_scores) if top1_scores else 0.0
        print(f"\n  [out-domain] avg top-1 BM25 score = {avg_ood:.4f}")
        new_rows.append({
            "chunker":         "bm25",
            "embedding_model": "-",
            "split":           "out_of_domain",
            "k":               "",
            "recall@k":        "",
            "precision@k":     "",
            "mrr":             "",
            "top1_sim_ood":    f"{avg_ood:.4f}",
        })

    # 6. Append vào CSV hiện có
    fieldnames = ["chunker", "embedding_model", "split", "k",
                  "recall@k", "precision@k", "mrr", "top1_sim_ood"]

    # Đọc các dòng cũ (nếu file đã tồn tại)
    existing_rows = []
    if output_csv.exists():
        with open(output_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
        # Xoá các dòng bm25 cũ (nếu có) để tránh duplicate
        existing_rows = [r for r in existing_rows if r.get("chunker") != "bm25"]

    all_rows = existing_rows + new_rows
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✅ Kết quả BM25 đã append vào {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunker", default="semantic", help="semantic | hierarchical")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent
    run_bm25_eval(base_path, chunker_name=args.chunker)
