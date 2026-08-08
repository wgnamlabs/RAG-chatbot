import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

# Reconfigure stdout for Windows console to handle Vietnamese characters
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Import đúng từ package embedding (không có file embedding.py, chỉ có embedder.py)
from indexing.embedding import SentenceTransformerEmbedder, MODELS_TO_COMPARE
from metrics import recall_at_k, precision_at_k, mean_reciprocal_rank


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity: (num_queries, num_chunks)."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def run_evaluation(chunks_dir: Path, questions_path: Path, output_csv: Path) -> None:
    if not chunks_dir.exists():
        print(f"Error: {chunks_dir} không tồn tại.")
        return

    with open(questions_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    questions     = [item["question"] for item in qa_data]
    ground_truths = [item["ground_truth_sources"] for item in qa_data]

    # Tách câu hỏi in-domain (có ground truth) và out-of-domain (không có)
    in_domain_idx  = [i for i, gt in enumerate(ground_truths) if gt]
    out_domain_idx = [i for i, gt in enumerate(ground_truths) if not gt]

    # NEW: gom câu hỏi in-domain theo source_doc_tag để tách báo cáo riêng
    # từng tài liệu, tránh việc tài liệu lớn (nhiều chunk) "che" điểm yếu
    # ở tài liệu nhỏ khi chỉ nhìn số liệu tổng gộp.
    idx_by_doc_tag = defaultdict(list)
    for i in in_domain_idx:
        tag = qa_data[i].get("source_doc_tag", "unknown")
        idx_by_doc_tag[tag].append(i)

    print(f"Tổng câu hỏi: {len(questions)} "
          f"(in-domain: {len(in_domain_idx)}, out-of-domain: {len(out_domain_idx)})")
    print(f"Phân bố câu hỏi in-domain theo tài liệu: "
          + ", ".join(f"{tag}={len(idxs)}" for tag, idxs in idx_by_doc_tag.items()))

    results_rows = []

    # Chạy tuần tự từng model để tiết kiệm VRAM
    for config in MODELS_TO_COMPARE:
        embedder = SentenceTransformerEmbedder(config)
        embedder.load()

        # is_query=True cho câu hỏi (bắt buộc cho Qwen3-Embedding)
        question_embeddings = embedder.encode(questions, is_query=True)

        for chunk_file in sorted(chunks_dir.glob("*.json")):
            chunker_name = chunk_file.stem
            print(f"\n{'='*60}")
            print(f"  {config.model_name}  ×  {chunker_name}")
            print(f"{'='*60}")

            with open(chunk_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            chunk_texts    = [c["text"]     for c in chunks]
            chunk_metadata = [c["metadata"] for c in chunks]

            # is_query=False cho chunk tài liệu
            chunk_embeddings = embedder.encode(chunk_texts, is_query=False)

            # Ma trận similarity: (num_questions, num_chunks)
            sim_matrix = cosine_similarity(question_embeddings, chunk_embeddings)

            k_values = [3, 5, 10, 15, 20, 30]

            # -------------------------------------------------------
            # (A) Đánh giá TỔNG GỘP tất cả câu hỏi in-domain (như cũ)
            # -------------------------------------------------------
            metrics_sum = {k: {"recall": 0.0, "precision": 0.0} for k in k_values}
            mrr_sum = 0.0

            for q_idx in in_domain_idx:
                scores = sim_matrix[q_idx]
                top_indices = np.argsort(scores)[::-1]
                retrieved_metadata = [chunk_metadata[i] for i in top_indices]

                for k in k_values:
                    metrics_sum[k]["recall"]    += recall_at_k(retrieved_metadata, ground_truths[q_idx], k)
                    metrics_sum[k]["precision"] += precision_at_k(retrieved_metadata, ground_truths[q_idx], k)
                mrr_sum += mean_reciprocal_rank(retrieved_metadata, ground_truths[q_idx])

            n_in = len(in_domain_idx)
            for k in k_values:
                avg_recall    = metrics_sum[k]["recall"]    / n_in
                avg_precision = metrics_sum[k]["precision"] / n_in
                avg_mrr       = mrr_sum / n_in

                results_rows.append({
                    "chunker":         chunker_name,
                    "embedding_model": config.model_name,
                    "split":           "in_domain",
                    "k":               k,
                    "recall@k":        f"{avg_recall:.4f}",
                    "precision@k":     f"{avg_precision:.4f}",
                    "mrr":             f"{avg_mrr:.4f}",
                    "top1_sim_ood":    "",
                })
                print(f"[in-domain TỔNG]  k={k}: Recall={avg_recall:.4f}, "
                      f"Precision={avg_precision:.4f}, MRR={avg_mrr:.4f}")

            # -------------------------------------------------------
            # (B) NEW: Đánh giá TÁCH RIÊNG theo từng tài liệu
            # Đây là con số quan trọng hơn để đánh giá chất lượng thật,
            # vì tài liệu lớn (nhiều chunk) dễ đạt điểm cao ngay cả khi
            # xếp hạng gần như ngẫu nhiên (xem baseline_random_eval.py).
            # -------------------------------------------------------
            for tag, idx_list in idx_by_doc_tag.items():
                n_tag = len(idx_list)
                for k in k_values:
                    recall_sum, precision_sum, mrr_tag_sum = 0.0, 0.0, 0.0
                    for q_idx in idx_list:
                        scores = sim_matrix[q_idx]
                        top_indices = np.argsort(scores)[::-1]
                        retrieved_metadata = [chunk_metadata[i] for i in top_indices]
                        gt = ground_truths[q_idx]
                        recall_sum    += recall_at_k(retrieved_metadata, gt, k)
                        precision_sum += precision_at_k(retrieved_metadata, gt, k)
                        mrr_tag_sum   += mean_reciprocal_rank(retrieved_metadata, gt)

                    results_rows.append({
                        "chunker":         chunker_name,
                        "embedding_model": config.model_name,
                        "split":           f"in_domain__{tag}",
                        "k":               k,
                        "recall@k":        f"{recall_sum/n_tag:.4f}",
                        "precision@k":     f"{precision_sum/n_tag:.4f}",
                        "mrr":             f"{mrr_tag_sum/n_tag:.4f}",
                        "top1_sim_ood":    "",
                    })
                print(f"  [{tag}] (n={n_tag}) k={k_values[-1]}: "
                      f"Recall={recall_sum/n_tag:.4f}, MRR={mrr_tag_sum/n_tag:.4f}")

            # -------------------------------------------------------
            # Đánh giá câu hỏi OUT-OF-DOMAIN (không có ground truth)
            # -------------------------------------------------------
            if out_domain_idx:
                top1_sims = []
                for q_idx in out_domain_idx:
                    top1_score = float(np.max(sim_matrix[q_idx]))
                    top1_sims.append(top1_score)

                avg_top1_ood = np.mean(top1_sims)
                results_rows.append({
                    "chunker":         chunker_name,
                    "embedding_model": config.model_name,
                    "split":           "out_of_domain",
                    "k":               "",
                    "recall@k":        "",
                    "precision@k":     "",
                    "mrr":             "",
                    "top1_sim_ood":    f"{avg_top1_ood:.4f}",
                })
                print(f"[out-domain] avg top-1 sim = {avg_top1_ood:.4f} "
                      f"(thấp hơn = phân biệt ngoài miền tốt hơn)")

        embedder.unload()

    # Ghi CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["chunker", "embedding_model", "split", "k",
                  "recall@k", "precision@k", "mrr", "top1_sim_ood"]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"\n✅ Evaluation complete! Results saved to {output_csv}")


if __name__ == "__main__":
    base_path      = Path(__file__).resolve().parent.parent
    chunks_dir     = base_path / "evaluation" / "results" / "chunks_cache"
    questions_path = base_path / "evaluation" / "eval_questions.json"
    output_csv     = base_path / "evaluation" / "results" / "comparison_matrix.csv"

    run_evaluation(chunks_dir, questions_path, output_csv)