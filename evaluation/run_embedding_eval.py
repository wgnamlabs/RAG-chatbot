from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# Windows console: giữ tiếng Việt.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Project root/src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from indexing.embedding import MODELS_TO_COMPARE, SentenceTransformerEmbedder
from metrics import (
    DEFAULT_EVIDENCE_MATCH,
    build_gold_relevance,
    complete_evidence_at_k,
    evidence_recall_at_k,
    hit_at_k,
    ndcg_at_k,
    ood_auroc,
    precision_at_k,
    reciprocal_rank,
)


CHUNK_FILES = {
    "semantic": "semantic.json",
    "hierarchical": "hierarchical.json",
}

K_VALUES = [3, 5, 10, 15, 20, 30]
PRIMARY_K = 10


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix: (num_queries, num_chunks)."""
    a_norms = np.linalg.norm(a, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)

    # Tránh chia 0 nếu model trả vector lỗi/zero.
    a_norms[a_norms == 0] = 1.0
    b_norms[b_norms == 0] = 1.0

    return (a / a_norms) @ (b / b_norms).T


def safe_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def fmt(value: float) -> str:
    if value is None or math.isnan(float(value)):
        return ""
    return f"{float(value):.4f}"


def load_eval_questions(path: Path, split: str) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        all_items = json.load(f)

    if split == "all":
        selected = all_items
    else:
        selected = [item for item in all_items if item.get("split") == split]

    if not selected:
        raise ValueError(f"Không có câu hỏi cho split={split!r}")

    ids = [item["id"] for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("eval_questions có ID trùng.")

    return selected


def load_chunks(chunks_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Chỉ đọc đúng 2 output chunker; KHÔNG glob chunking_summary.json."""
    result = {}
    for chunker_name, filename in CHUNK_FILES.items():
        path = chunks_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Thiếu chunk file: {path}")

        with path.open("r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not isinstance(chunks, list):
            raise ValueError(f"{path} không phải list chunks.")

        for i, chunk in enumerate(chunks):
            if "text" not in chunk or "metadata" not in chunk:
                raise ValueError(f"{path}: chunk {i} thiếu text/metadata.")

        result[chunker_name] = chunks

    return result


def build_all_gold_maps(
    qa_data: Sequence[Dict[str, Any]],
    chunks_by_chunker: Dict[str, List[Dict[str, Any]]],
) -> Tuple[
    Dict[str, Dict[str, Dict[str, Any]]],
    Dict[str, Any],
]:
    """Map gold evidence -> chunks trước khi load embedding models.

    Nếu bất kỳ evidence nào không map được ở một chunker, benchmark dừng:
    không âm thầm chấm sai.
    """
    all_maps: Dict[str, Dict[str, Dict[str, Any]]] = {}
    report: Dict[str, Any] = {
        "match_config": {
            "min_coverage": DEFAULT_EVIDENCE_MATCH.min_coverage,
            "min_contiguous_tokens": DEFAULT_EVIDENCE_MATCH.min_contiguous_tokens,
        },
        "chunkers": {},
    }

    for chunker_name, chunks in chunks_by_chunker.items():
        maps_for_chunker = {}
        unmapped_rows = []
        evidence_total = 0
        relevant_counts = []

        for item in qa_data:
            if not item.get("answerable", False):
                continue

            gold_map = build_gold_relevance(item, chunks)
            maps_for_chunker[item["id"]] = gold_map

            for evidence_id, chunk_indices in gold_map["evidence_to_chunks"].items():
                evidence_total += 1
                relevant_counts.append(len(chunk_indices))

            for evidence_id in gold_map["unmapped_evidence"]:
                unmapped_rows.append({
                    "question_id": item["id"],
                    "evidence_id": evidence_id,
                    "best_candidate": gold_map["best_candidates"].get(evidence_id),
                })

        report["chunkers"][chunker_name] = {
            "evidence_total": evidence_total,
            "unmapped_evidence_count": len(unmapped_rows),
            "avg_relevant_chunks_per_evidence": (
                float(np.mean(relevant_counts)) if relevant_counts else 0.0
            ),
            "min_relevant_chunks_per_evidence": (
                min(relevant_counts) if relevant_counts else 0
            ),
            "max_relevant_chunks_per_evidence": (
                max(relevant_counts) if relevant_counts else 0
            ),
            "unmapped": unmapped_rows,
        }

        if unmapped_rows:
            first = unmapped_rows[0]
            raise RuntimeError(
                f"[{chunker_name}] Có {len(unmapped_rows)} gold evidence không map "
                f"được sang chunks. Ví dụ: {first}. "
                "Không nên chạy embedding benchmark khi gold mapping chưa sạch."
            )

        all_maps[chunker_name] = maps_for_chunker

    return all_maps, report


def question_metrics(
    ranking: Sequence[int],
    gold_map: Dict[str, Any],
    k_values: Sequence[int],
) -> Dict[str, float]:
    out: Dict[str, float] = {
        "mrr": reciprocal_rank(ranking, gold_map),
    }

    for k in k_values:
        out[f"recall@{k}"] = evidence_recall_at_k(ranking, gold_map, k)
        out[f"precision@{k}"] = precision_at_k(ranking, gold_map, k)
        out[f"hit@{k}"] = hit_at_k(ranking, gold_map, k)
        out[f"complete@{k}"] = complete_evidence_at_k(ranking, gold_map, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranking, gold_map, k)

    return out


def aggregate_question_rows(
    rows: Sequence[Dict[str, Any]],
    k_values: Sequence[int],
) -> Dict[str, float]:
    if not rows:
        return {}

    summary: Dict[str, float] = {
        "n_questions": len(rows),
        "mrr": safe_mean([r["mrr"] for r in rows]),
    }

    for k in k_values:
        for metric in ("recall", "precision", "hit", "complete", "ndcg"):
            key = f"{metric}@{k}"
            summary[key] = safe_mean([r[key] for r in rows])

    return summary


def append_group_rows(
    output: List[Dict[str, Any]],
    question_rows: Sequence[Dict[str, Any]],
    *,
    chunker: str,
    model_name: str,
    eval_split: str,
) -> None:
    in_domain = [r for r in question_rows if r["answerable"]]

    groups: List[Tuple[str, str, List[Dict[str, Any]]]] = [
        ("overall", "in_domain", in_domain),
    ]

    # Per-document
    by_doc = defaultdict(list)
    for row in in_domain:
        by_doc[row["source_doc_tag"]].append(row)
    groups.extend(
        ("document", tag, rows)
        for tag, rows in sorted(by_doc.items())
    )

    # Standard vs hard_multi
    by_benchmark = defaultdict(list)
    for row in in_domain:
        by_benchmark[row["benchmark_group"]].append(row)
    groups.extend(
        ("benchmark_group", group, rows)
        for group, rows in sorted(by_benchmark.items())
    )

    # Các slice quan trọng cho corpus này.
    table_rows = [r for r in in_domain if r["requires_table"]]
    patient_rows = [
        r for r in in_domain
        if r["question_style"] == "patient_style"
        or r["question_type"] == "patient_style"
    ]
    numeric_rows = [
        r for r in in_domain
        if r["base_question_type"] == "numeric"
        or r["question_type"] == "numeric"
    ]
    if table_rows:
        groups.append(("slice", "requires_table", table_rows))
    if patient_rows:
        groups.append(("slice", "patient_style", patient_rows))
    if numeric_rows:
        groups.append(("slice", "numeric", numeric_rows))

    for group_type, group_value, rows in groups:
        metrics = aggregate_question_rows(rows, K_VALUES)
        if not metrics:
            continue

        for k in K_VALUES:
            output.append({
                "chunker": chunker,
                "embedding_model": model_name,
                "eval_split": eval_split,
                "group_type": group_type,
                "group_value": group_value,
                "n_questions": metrics["n_questions"],
                "k": k,
                "recall@k": metrics[f"recall@{k}"],
                "precision@k": metrics[f"precision@{k}"],
                "hit@k": metrics[f"hit@{k}"],
                "evidence_complete@k": metrics[f"complete@{k}"],
                "ndcg@k": metrics[f"ndcg@{k}"],
                "mrr": metrics["mrr"],
            })


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_evaluation(
    chunks_dir: Path,
    questions_path: Path,
    output_dir: Path,
    eval_split: str,
) -> None:
    qa_data = load_eval_questions(questions_path, eval_split)
    chunks_by_chunker = load_chunks(chunks_dir)

    in_domain = [x for x in qa_data if x.get("answerable", False)]
    ood = [x for x in qa_data if not x.get("answerable", False)]

    print("=" * 72)
    print("MASTER EMBEDDING / CHUNKING RETRIEVAL EVALUATION")
    print("=" * 72)
    print(f"Split      : {eval_split}")
    print(f"Questions  : {len(qa_data)}")
    print(f"In-domain  : {len(in_domain)}")
    print(f"OOD        : {len(ood)}")
    print(f"Chunkers   : {', '.join(CHUNK_FILES)}")
    print(f"K values   : {K_VALUES}")
    print()

    # Ground-truth mapping là độc lập embedding model -> làm 1 lần.
    gold_maps, mapping_report = build_all_gold_maps(
        qa_data,
        chunks_by_chunker,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / f"evidence_mapping_report_{eval_split}.json"
    mapping_path.write_text(
        json.dumps(mapping_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Gold-evidence mapping:")
    for chunker, info in mapping_report["chunkers"].items():
        print(
            f"  {chunker}: evidence={info['evidence_total']}, "
            f"unmapped={info['unmapped_evidence_count']}, "
            f"avg relevant chunks/evidence="
            f"{info['avg_relevant_chunks_per_evidence']:.2f}"
        )
    print()

    questions = [item["question"] for item in qa_data]

    all_question_rows: List[Dict[str, Any]] = []
    all_group_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    summary_json: Dict[str, Any] = {
        "eval_split": eval_split,
        "n_questions": len(qa_data),
        "n_in_domain": len(in_domain),
        "n_ood": len(ood),
        "k_values": K_VALUES,
        "primary_k": PRIMARY_K,
        "evidence_match": mapping_report["match_config"],
        "experiments": [],
    }

    for model_config in MODELS_TO_COMPARE:
        embedder = SentenceTransformerEmbedder(model_config)
        embedder.load()

        try:
            # Qwen3 nhận query instruction ở đây; docs/chunks không nhận query prompt.
            query_embeddings = embedder.encode(questions, is_query=True)

            for chunker_name, chunks in chunks_by_chunker.items():
                print("\n" + "=" * 72)
                print(f"{chunker_name.upper()} × {model_config.model_name}")
                print("=" * 72)

                chunk_texts = [chunk["text"] for chunk in chunks]
                chunk_embeddings = embedder.encode(chunk_texts, is_query=False)
                sim_matrix = cosine_similarity(
                    query_embeddings,
                    chunk_embeddings,
                )

                question_rows_for_combo: List[Dict[str, Any]] = []
                id_top1_sims: List[float] = []
                ood_top1_sims: List[float] = []

                for q_pos, item in enumerate(qa_data):
                    scores = sim_matrix[q_pos]
                    ranking = np.argsort(scores)[::-1].tolist()
                    top_idx = ranking[0]
                    top1_sim = float(scores[top_idx])
                    top_chunk = chunks[top_idx]

                    base_row = {
                        "question_id": item["id"],
                        "question": item["question"],
                        "chunker": chunker_name,
                        "embedding_model": model_config.model_name,
                        "eval_split": eval_split,
                        "answerable": bool(item["answerable"]),
                        "benchmark_group": item["benchmark_group"],
                        "difficulty": item["difficulty"],
                        "question_type": item["question_type"],
                        "base_question_type": item["base_question_type"],
                        "question_style": item["question_style"],
                        "requires_table": bool(item["requires_table"]),
                        "requires_multiple_evidence": bool(
                            item["requires_multiple_evidence"]
                        ),
                        "source_doc_tag": item["source_doc_tag"] or "",
                        "gold_evidence_count": item["gold_evidence_count"],
                        "top1_similarity": top1_sim,
                        "top_chunk_id": top_chunk["metadata"].get("chunk_id", ""),
                        "top_chunk_source": top_chunk["metadata"].get("source", ""),
                    }

                    if item["answerable"]:
                        id_top1_sims.append(top1_sim)
                        gold_map = gold_maps[chunker_name][item["id"]]
                        q_metrics = question_metrics(ranking, gold_map, K_VALUES)
                        base_row.update(q_metrics)
                    else:
                        ood_top1_sims.append(top1_sim)
                        base_row["mrr"] = ""
                        for k in K_VALUES:
                            for metric in (
                                "recall",
                                "precision",
                                "hit",
                                "complete",
                                "ndcg",
                            ):
                                base_row[f"{metric}@{k}"] = ""

                    question_rows_for_combo.append(base_row)
                    all_question_rows.append(base_row)

                # Aggregate in-domain + important slices.
                append_group_rows(
                    all_group_rows,
                    question_rows_for_combo,
                    chunker=chunker_name,
                    model_name=model_config.model_name,
                    eval_split=eval_split,
                )

                in_rows = [
                    r for r in question_rows_for_combo if r["answerable"]
                ]
                overall = aggregate_question_rows(in_rows, K_VALUES)

                mean_id = safe_mean(id_top1_sims)
                mean_ood = safe_mean(ood_top1_sims)
                margin = (
                    mean_id - mean_ood
                    if not math.isnan(mean_id) and not math.isnan(mean_ood)
                    else float("nan")
                )
                auroc = ood_auroc(id_top1_sims, ood_top1_sims)

                comparison = {
                    "chunker": chunker_name,
                    "embedding_model": model_config.model_name,
                    "eval_split": eval_split,
                    "n_in_domain": len(in_rows),
                    "n_ood": len(ood_top1_sims),
                    f"recall@{PRIMARY_K}": overall[f"recall@{PRIMARY_K}"],
                    f"precision@{PRIMARY_K}": overall[f"precision@{PRIMARY_K}"],
                    f"hit@{PRIMARY_K}": overall[f"hit@{PRIMARY_K}"],
                    f"evidence_complete@{PRIMARY_K}": overall[
                        f"complete@{PRIMARY_K}"
                    ],
                    f"ndcg@{PRIMARY_K}": overall[f"ndcg@{PRIMARY_K}"],
                    "mrr": overall["mrr"],
                    "top1_sim_id": mean_id,
                    "top1_sim_ood": mean_ood,
                    "id_ood_margin": margin,
                    "ood_auroc": auroc,
                }
                comparison_rows.append(comparison)

                summary_json["experiments"].append({
                    **comparison,
                    "model_note": model_config.note,
                })

                print(
                    f"Recall@{PRIMARY_K}={overall[f'recall@{PRIMARY_K}']:.4f} | "
                    f"Precision@{PRIMARY_K}="
                    f"{overall[f'precision@{PRIMARY_K}']:.4f} | "
                    f"MRR={overall['mrr']:.4f} | "
                    f"nDCG@{PRIMARY_K}={overall[f'ndcg@{PRIMARY_K}']:.4f}"
                )
                if ood_top1_sims:
                    print(
                        f"OOD: top1 ID={mean_id:.4f}, OOD={mean_ood:.4f}, "
                        f"margin={margin:.4f}, AUROC={auroc:.4f}"
                    )

        finally:
            embedder.unload()

    # ---------------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------------
    comparison_path = (
        output_dir.parent / f"comparison_matrix_{eval_split}.csv"
    )
    comparison_fields = [
        "chunker",
        "embedding_model",
        "eval_split",
        "n_in_domain",
        "n_ood",
        f"recall@{PRIMARY_K}",
        f"precision@{PRIMARY_K}",
        f"hit@{PRIMARY_K}",
        f"evidence_complete@{PRIMARY_K}",
        f"ndcg@{PRIMARY_K}",
        "mrr",
        "top1_sim_id",
        "top1_sim_ood",
        "id_ood_margin",
        "ood_auroc",
    ]
    write_csv(comparison_path, comparison_rows, comparison_fields)

    per_question_path = output_dir / f"per_question_results_{eval_split}.csv"
    per_question_fields = [
        "question_id",
        "question",
        "chunker",
        "embedding_model",
        "eval_split",
        "answerable",
        "benchmark_group",
        "difficulty",
        "question_type",
        "base_question_type",
        "question_style",
        "requires_table",
        "requires_multiple_evidence",
        "source_doc_tag",
        "gold_evidence_count",
        "top1_similarity",
        "top_chunk_id",
        "top_chunk_source",
        "mrr",
    ]
    for k in K_VALUES:
        per_question_fields.extend([
            f"recall@{k}",
            f"precision@{k}",
            f"hit@{k}",
            f"complete@{k}",
            f"ndcg@{k}",
        ])
    write_csv(
        per_question_path,
        all_question_rows,
        per_question_fields,
    )

    per_group_path = output_dir / f"per_group_results_{eval_split}.csv"
    group_fields = [
        "chunker",
        "embedding_model",
        "eval_split",
        "group_type",
        "group_value",
        "n_questions",
        "k",
        "recall@k",
        "precision@k",
        "hit@k",
        "evidence_complete@k",
        "ndcg@k",
        "mrr",
    ]
    write_csv(per_group_path, all_group_rows, group_fields)

    summary_path = output_dir / f"embedding_summary_{eval_split}.json"
    summary_path.write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Comparison       : {comparison_path}")
    print(f"Per-question     : {per_question_path}")
    print(f"Per-group        : {per_group_path}")
    print(f"Summary JSON     : {summary_path}")
    print(f"Evidence mapping : {mapping_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 2 chunkers × 3 embedding models bằng master eval v2. "
            "Mặc định chỉ chạy DEV để tránh dùng TEST khi tuning."
        )
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        default="dev",
        help="dev để chọn config; test chỉ chạy sau khi đã chốt config.",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "chunks_cache",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "eval_questions.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "embedding_eval",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        chunks_dir=args.chunks_dir,
        questions_path=args.questions,
        output_dir=args.output_dir,
        eval_split=args.split,
    )
