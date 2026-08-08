"""
Tính baseline "xếp hạng ngẫu nhiên" (không dùng embedding, không dùng AI)
để so sánh với kết quả model thật trong comparison_matrix.csv.

Ý nghĩa: nếu recall/MRR của model chỉ nhỉnh hơn chút xíu so với baseline
này, nghĩa là phần lớn điểm số đến từ corpus bị lệch (1 tài liệu chiếm
đa số chunk), KHÔNG phải nhờ model "hiểu" nội dung. Baseline càng thấp mà
model càng cao thì model càng thực sự có giá trị.

Cách chạy: giống hệt run_embedding_eval.py nhưng KHÔNG encode gì cả —
chỉ xáo trộn ngẫu nhiên thứ tự chunk nhiều lần rồi lấy trung bình.
"""

import json
import csv
import random
from pathlib import Path
from collections import defaultdict
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics import recall_at_k, precision_at_k, mean_reciprocal_rank

N_TRIALS = 20  # số lần xáo trộn ngẫu nhiên để lấy trung bình ổn định


def run_random_baseline(chunks_dir: Path, questions_path: Path, output_csv: Path) -> None:
    with open(questions_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    ground_truths = [item["ground_truth_sources"] for item in qa_data]
    in_domain_idx = [i for i, gt in enumerate(ground_truths) if gt]

    idx_by_doc_tag = defaultdict(list)
    for i in in_domain_idx:
        tag = qa_data[i].get("source_doc_tag", "unknown")
        idx_by_doc_tag[tag].append(i)

    results_rows = []
    k_values = [3, 5, 10, 15, 20, 30]

    for chunk_file in sorted(chunks_dir.glob("*.json")):
        chunker_name = chunk_file.stem
        print(f"\n=== Baseline ngẫu nhiên × {chunker_name} ===")

        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        chunk_metadata = [c["metadata"] for c in chunks]
        n_chunks = len(chunk_metadata)

        # (A) Baseline TỔNG GỘP
        for k in k_values:
            recall_trials, precision_trials, mrr_trials = [], [], []
            for _ in range(N_TRIALS):
                order = list(range(n_chunks))
                random.shuffle(order)
                shuffled_meta = [chunk_metadata[i] for i in order]

                r_sum = p_sum = m_sum = 0.0
                for q_idx in in_domain_idx:
                    gt = ground_truths[q_idx]
                    r_sum += recall_at_k(shuffled_meta, gt, k)
                    p_sum += precision_at_k(shuffled_meta, gt, k)
                    m_sum += mean_reciprocal_rank(shuffled_meta, gt)
                n = len(in_domain_idx)
                recall_trials.append(r_sum / n)
                precision_trials.append(p_sum / n)
                mrr_trials.append(m_sum / n)

            avg_recall = sum(recall_trials) / N_TRIALS
            avg_precision = sum(precision_trials) / N_TRIALS
            avg_mrr = sum(mrr_trials) / N_TRIALS

            results_rows.append({
                "chunker": chunker_name, "embedding_model": "RANDOM_BASELINE",
                "split": "in_domain", "k": k,
                "recall@k": f"{avg_recall:.4f}",
                "precision@k": f"{avg_precision:.4f}",
                "mrr": f"{avg_mrr:.4f}", "top1_sim_ood": "",
            })
            print(f"[TỔNG]     k={k}: Recall={avg_recall:.4f}, MRR={avg_mrr:.4f}")

        # (B) Baseline TÁCH RIÊNG theo tài liệu
        for tag, idx_list in idx_by_doc_tag.items():
            n_tag = len(idx_list)
            for k in k_values:
                recall_trials, mrr_trials = [], []
                for _ in range(N_TRIALS):
                    order = list(range(n_chunks))
                    random.shuffle(order)
                    shuffled_meta = [chunk_metadata[i] for i in order]
                    r_sum = m_sum = 0.0
                    for q_idx in idx_list:
                        gt = ground_truths[q_idx]
                        r_sum += recall_at_k(shuffled_meta, gt, k)
                        m_sum += mean_reciprocal_rank(shuffled_meta, gt)
                    recall_trials.append(r_sum / n_tag)
                    mrr_trials.append(m_sum / n_tag)

                avg_recall = sum(recall_trials) / N_TRIALS
                avg_mrr = sum(mrr_trials) / N_TRIALS
                results_rows.append({
                    "chunker": chunker_name, "embedding_model": "RANDOM_BASELINE",
                    "split": f"in_domain__{tag}", "k": k,
                    "recall@k": f"{avg_recall:.4f}", "precision@k": "",
                    "mrr": f"{avg_mrr:.4f}", "top1_sim_ood": "",
                })
            print(f"  [{tag}] k={k_values[-1]}: Recall={avg_recall:.4f}, MRR={avg_mrr:.4f}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["chunker", "embedding_model", "split", "k",
                  "recall@k", "precision@k", "mrr", "top1_sim_ood"]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"\n✅ Baseline ngẫu nhiên đã lưu → {output_csv}")
    print("So sánh file này với comparison_matrix.csv để thấy model 'thắng' baseline bao nhiêu.")


if __name__ == "__main__":
    base_path = Path(__file__).resolve().parent.parent
    chunks_dir = base_path / "evaluation" / "results" / "chunks_cache"
    questions_path = base_path / "evaluation" / "eval_questions.json"
    output_csv = base_path / "evaluation" / "results" / "random_baseline.csv"

    run_random_baseline(chunks_dir, questions_path, output_csv)