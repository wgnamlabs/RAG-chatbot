"""
eval_pipeline.py — So sánh 3 cấu hình retrieval trên 82 câu hỏi benchmark.

Chạy:
    python evaluation/eval_pipeline.py
    python evaluation/eval_pipeline.py --device cuda   # nếu có GPU

3 cấu hình:
    (a) Dense only       — dense_search(top_k=10), không rerank
    (b) Hybrid RRF       — hybrid_search(top_k=10), không rerank
    (c) Hybrid + Rerank  — hybrid_search(top_k=15) → rerank(top_k=10) → dedup(final_top_k=5)

Metrics in-domain (72 câu):
    Recall@5, Recall@10, MRR — tính trên unique sources (dedup trước khi so sánh)

OOD (10 câu):
    Chạy run_pipeline() đầy đủ, kiểm tra answer có "Không tìm thấy thông tin"

Load model 1 LẦN duy nhất ngoài vòng lặp.

Output: in ra console + lưu evaluation/results/results_comparison.md
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import unicodedata
import numpy as np
from pathlib import Path

# ── sys.path setup ────────────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# ── Imports ───────────────────────────────────────────────────────────────────
from indexing.embedding import SentenceTransformerEmbedder
from indexing.embedding.config import EmbedderConfig
from indexing.vector_store import QdrantVectorStore, QdrantStoreConfig

from generation.schemas import Chunk
from generation.pipeline.retrieval import dense_search, hybrid_search
from generation.pipeline.rerank import rerank as rerank_fn
from generation.pipeline.postprocess import dedup_redundant
from generation.pipeline.main import run_pipeline, DEFAULT_CONFIG


# ── Metrics helpers (wrapper cho metrics.py, làm việc với Chunk list) ─────────

def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)

def _unique_sources(chunks: list[Chunk]) -> list[str]:
    """Dedup source giữ thứ tự — KHÔNG đếm nhiều chunk cùng file là nhiều hit."""
    return list(dict.fromkeys(_nfc(c.source) for c in chunks))

def recall_at_k_chunks(chunks: list[Chunk], gt_sources: list[str], k: int) -> float:
    """Recall@k từ list[Chunk], dedup source trước."""
    if not gt_sources:
        return 0.0
    gt_set = {_nfc(s) for s in gt_sources}
    retrieved = _unique_sources(chunks)[:k]
    hits = sum(1 for s in retrieved if s in gt_set)
    return hits / len(gt_set)

def precision_at_k_chunks(chunks: list[Chunk], gt_sources: list[str], k: int) -> float:
    """Precision@k từ list[Chunk], đo ở mức độ chunk, không dedup source."""
    if not gt_sources:
        return 0.0
    gt_set = {_nfc(s) for s in gt_sources}
    retrieved_k = chunks[:k]
    if not retrieved_k:
        return 0.0
    hits = sum(1 for c in retrieved_k if _nfc(c.source) in gt_set)
    return hits / len(retrieved_k)

def mrr_chunks(chunks: list[Chunk], gt_sources: list[str]) -> float:
    """MRR từ list[Chunk], dedup source trước."""
    if not gt_sources:
        return 0.0
    gt_set = {_nfc(s) for s in gt_sources}
    for rank_one, source in enumerate(_unique_sources(chunks), start=1):
        if source in gt_set:
            return 1.0 / rank_one
    return 0.0


# ── Load hạ tầng ──────────────────────────────────────────────────────────────

def load_infra(device: str = "cpu") -> tuple:
    """Load embedder, Qdrant, BM25. Trả về (embedder, qdrant_store, bm25_data)."""
    print(f"[Setup] Load Vietnamese_Embedding_v2 trên {device}...")
    emb_config = EmbedderConfig(
        model_name="AITeamVN/Vietnamese_Embedding_v2",
        batch_size=32,
        max_seq_length=2048,
        device=device,
    )
    embedder = SentenceTransformerEmbedder(emb_config)
    embedder.load()

    print("[Setup] Kết nối Qdrant localhost:6333...")
    qdrant_store = QdrantVectorStore(config=QdrantStoreConfig(collection_name="phu_san_chunks"))
    qdrant_store.load()

    bm25_path = BASE / "data" / "vector_db" / "bm25_index.pkl"
    print(f"[Setup] Load BM25 từ {bm25_path}...")
    with open(bm25_path, "rb") as f:
        bm25_data = pickle.load(f)
    print(f"  → {len(bm25_data['chunks'])} chunks trong BM25 index")

    return embedder, qdrant_store, bm25_data


def load_reranker(device: str = "cpu"):
    """Load CrossEncoder 1 lần duy nhất để dùng suốt eval."""
    from sentence_transformers import CrossEncoder
    print("[Setup] Load CrossEncoder bge-reranker-v2-m3...")
    return CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)


# ── Eval in-domain (3 cấu hình) ──────────────────────────────────────────────

def eval_config(
    name: str,
    questions: list[str],
    gt_sources_list: list[list[str]],
    in_domain_idx: list[int],
    retrieval_fn,           # callable(query) → list[Chunk]
    k_values: list[int] = (5, 10),
) -> dict:
    """Chạy eval 1 cấu hình, trả về dict metrics."""
    recall_sums = {k: 0.0 for k in k_values}
    precision_sums = {k: 0.0 for k in k_values}
    mrr_sum = 0.0
    latencies = []
    n = len(in_domain_idx)

    print(f"\n[{name}] Đang eval {n} câu in-domain...")
    for i, q_idx in enumerate(in_domain_idx):
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n}...")

        gt = gt_sources_list[q_idx]
        t0 = time.perf_counter()
        chunks = retrieval_fn(questions[q_idx])
        latencies.append(time.perf_counter() - t0)

        for k in k_values:
            recall_sums[k] += recall_at_k_chunks(chunks, gt, k)
            precision_sums[k] += precision_at_k_chunks(chunks, gt, k)
        mrr_sum += mrr_chunks(chunks, gt)

    return {
        "name":        name,
        "n":           n,
        **{f"Recall@{k}": recall_sums[k] / n for k in k_values},
        **{f"Precision@{k}": precision_sums[k] / n for k in k_values},
        "MRR":         mrr_sum / n,
        "latency_avg": np.mean(latencies),
        "latency_p50": np.percentile(latencies, 50),
        "latency_p95": np.percentile(latencies, 95),
        "latency_p99": np.percentile(latencies, 99),
    }


# ── Eval OOD ──────────────────────────────────────────────────────────────────

def eval_ood(
    questions: list[str],
    ood_idx: list[int],
    embedder,
    qdrant_store,
    bm25_data: dict,
    pipeline_config: dict,
) -> dict:
    """Chạy run_pipeline() đầy đủ trên 10 câu OOD, tính % từ chối đúng."""
    n = len(ood_idx)
    refused = 0

    print(f"\n[OOD] Đang eval {n} câu out-of-domain qua full pipeline...")
    for i, q_idx in enumerate(ood_idx):
        q = questions[q_idx]
        print(f"  [{i + 1}/{n}] {q[:60]}...")
        try:
            result = run_pipeline(
                query=q,
                config=pipeline_config,
                qdrant_store=qdrant_store,
                bm25_data=bm25_data,
                embedder=embedder,
            )
            answer = result.answer
        except Exception as exc:
            print(f"    ⚠️  Pipeline lỗi: {exc}")
            answer = ""

        refused_correctly = "Không tìm thấy thông tin" in answer
        if refused_correctly:
            refused += 1
            print(f"    ✓ Từ chối đúng")
        else:
            print(f"    ✗ Không từ chối — answer: {answer[:80]}...")

    return {"n_ood": n, "refused": refused, "pct": refused / n * 100 if n else 0}


# ── Format output ─────────────────────────────────────────────────────────────

def format_markdown_table(results: list[dict], ood_result: dict) -> str:
    """Tạo bảng markdown so sánh 3 cấu hình."""
    lines = [
        "# Kết quả Eval Pipeline — RAG Phụ Sản\n",
        "## So sánh 3 cấu hình Retrieval\n",
        "| Cấu hình | R@5 | R@10 | P@5 | P@10 | MRR | Latency (Avg / P50 / P95) |",
        "|----------|-----|------|-----|------|-----|---------------------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} "
            f"| {r.get('Recall@5', 0):.4f} "
            f"| {r.get('Recall@10', 0):.4f} "
            f"| {r.get('Precision@5', 0):.4f} "
            f"| {r.get('Precision@10', 0):.4f} "
            f"| {r['MRR']:.4f} "
            f"| {r['latency_avg']:.3f}s / {r['latency_p50']:.3f}s / {r['latency_p95']:.3f}s |"
        )
    lines.append("")
    lines.append("## Out-of-Domain: Khả năng từ chối câu hỏi ngoài miền\n")
    lines.append(
        f"OOD từ chối đúng: **{ood_result['refused']}/{ood_result['n_ood']} "
        f"({ood_result['pct']:.0f}%)**"
    )
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(device: str = "cpu", skip_ood: bool = False) -> None:
    # ── Load eval set ─────────────────────────────────────────────────────────
    questions_path = BASE / "evaluation" / "eval_questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    questions     = [item["question"]             for item in qa_data]
    gt_sources    = [item["ground_truth_sources"]  for item in qa_data]
    in_domain_idx = [i for i, gt in enumerate(gt_sources) if gt]
    ood_idx       = [i for i, gt in enumerate(gt_sources) if not gt]

    print(f"\nEval set: {len(questions)} câu ({len(in_domain_idx)} in-domain, {len(ood_idx)} OOD)")

    # ── Load infra (1 lần duy nhất) ───────────────────────────────────────────
    embedder, qdrant_store, bm25_data = load_infra(device=device)
    cross_encoder = load_reranker(device=device)

    # ── Định nghĩa 3 retrieval functions ─────────────────────────────────────
    # (a) Dense only — top_k=10
    def dense_only(q: str) -> list[Chunk]:
        return dense_search(q, qdrant_store, embedder, top_k=10)

    # (b) Hybrid RRF — top_k=10, không rerank
    def hybrid_only(q: str) -> list[Chunk]:
        return hybrid_search(q, q, qdrant_store, bm25_data, embedder, top_k=10)

    # (c) Hybrid + Rerank + Dedup — trả về top 10 cuối cùng
    def hybrid_rerank(q: str) -> list[Chunk]:
        chunks_15 = hybrid_search(q, q, qdrant_store, bm25_data, embedder, top_k=15)
        if not chunks_15:
            return []
        from sentence_transformers import CrossEncoder
        # Dùng lại cross_encoder đã load — gọi predict trực tiếp
        pairs  = [(q, c.text) for c in chunks_15]
        scores = cross_encoder.predict(pairs).tolist()
        reranked = [
            c.model_copy(update={"score": float(s)})
            for c, s in zip(chunks_15, scores)
        ]
        reranked.sort(key=lambda c: c.score, reverse=True)
        top10 = reranked[:10]
        # QUAN TRỌNG: để eval Recall@10 công bằng, ta lấy top 10 thay vì 5 như trong pipeline chạy thật
        deduped = dedup_redundant(top10, embedder, sim_threshold=0.9, final_top_k=10)
        return deduped

    # ── Chạy eval 3 cấu hình ─────────────────────────────────────────────────
    all_results = []
    for cfg_name, fn in [
        ("Dense only",      dense_only),
        ("Hybrid RRF",      hybrid_only),
        ("Hybrid + Rerank", hybrid_rerank),
    ]:
        result = eval_config(
            name=cfg_name,
            questions=questions,
            gt_sources_list=gt_sources,
            in_domain_idx=in_domain_idx,
            retrieval_fn=fn,
            k_values=[5, 10],
        )
        all_results.append(result)

        # In ngay kết quả từng cấu hình
        print(f"\n  → Recall@5={result['Recall@5']:.4f} | Recall@10={result['Recall@10']:.4f}")
        print(f"  → Precision@5={result['Precision@5']:.4f} | Precision@10={result['Precision@10']:.4f}")
        print(f"  → MRR={result['MRR']:.4f} | Latency={result['latency_avg']:.3f}s (P95={result['latency_p95']:.3f}s)")

    # ── Eval OOD ──────────────────────────────────────────────────────────────
    ood_result = {"n_ood": len(ood_idx), "refused": 0, "pct": 0.0}
    if not skip_ood and ood_idx:
        pipeline_cfg = {
            **DEFAULT_CONFIG,
            "rerank_device": device,
        }
        ood_result = eval_ood(
            questions=questions,
            ood_idx=ood_idx,
            embedder=embedder,
            qdrant_store=qdrant_store,
            bm25_data=bm25_data,
            pipeline_config=pipeline_cfg,
        )

    # ── In bảng kết quả ───────────────────────────────────────────────────────
    md_content = format_markdown_table(all_results, ood_result)
    print("\n" + "=" * 60)
    print(md_content)
    print("=" * 60)

    # ── Lưu file ──────────────────────────────────────────────────────────────
    out_path = BASE / "evaluation" / "results" / "results_comparison.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_content, encoding="utf-8")
    print(f"\nĐã lưu kết quả → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval RAG pipeline — 3 cấu hình so sánh")
    parser.add_argument("--device",   default="cpu",   help="cuda | cpu (default: cpu)")
    parser.add_argument("--skip-ood", action="store_true", help="Bỏ qua phần eval OOD")
    args = parser.parse_args()

    main(device=args.device, skip_ood=args.skip_ood)
