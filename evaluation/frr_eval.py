"""
frr_eval.py — Đo False Rejection Rate (FRR) trên tập in-domain.

FRR = tỷ lệ câu hỏi đúng domain bị pipeline từ chối oan
     ("Không tìm thấy thông tin..." khi lẽ ra phải trả lời)

Báo cáo riêng 2 nhóm (nếu có field "boundary_type" trong eval_questions.json):
    - clear   : câu hỏi rõ ràng trong domain
    - boundary: câu hỏi mơ hồ / ranh giới

Chạy:
    python evaluation/frr_eval.py --device cuda

Output: evaluation/results/frr_results.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import logging
logging.basicConfig(level=logging.WARNING)

from indexing.embedding import SentenceTransformerEmbedder
from indexing.embedding.config import EmbedderConfig
from indexing.vector_store import QdrantVectorStore, QdrantStoreConfig
from generation.pipeline.main import run_pipeline, DEFAULT_CONFIG

# Từ/cụm báo hiệu từ chối (giữ đồng bộ với _NO_CONTEXT_ANSWER trong main.py)
REJECTION_SIGNALS = [
    "không tìm thấy thông tin",
    "ngoài phạm vi",
    "không có thông tin",
]


def is_rejected(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(sig in answer_lower for sig in REJECTION_SIGNALS)


def main(device: str = "cuda") -> None:
    # ── Load eval set ─────────────────────────────────────────────────────────
    with open(BASE / "evaluation" / "eval_questions.json", encoding="utf-8") as f:
        qa_data = json.load(f)

    in_domain = [x for x in qa_data if x.get("ground_truth_sources")]
    print(f"\nEval FRR trên {len(in_domain)} câu in-domain...\n")

    # ── Load infra ────────────────────────────────────────────────────────────
    print("[Setup] Load infra...")
    embedder = SentenceTransformerEmbedder(EmbedderConfig(
        model_name="AITeamVN/Vietnamese_Embedding_v2",
        batch_size=32, max_seq_length=2048, device=device,
    ))
    embedder.load()
    qdrant_store = QdrantVectorStore(config=QdrantStoreConfig(collection_name="phu_san_chunks"))
    qdrant_store.load()
    with open(BASE / "data" / "vector_db" / "bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)

    cfg = {**DEFAULT_CONFIG, "rerank_device": device}

    # ── Chạy pipeline ─────────────────────────────────────────────────────────
    results = []
    for i, item in enumerate(in_domain):
        q = item["question"]
        tag = item.get("source_doc_tag", "unknown")
        # "boundary_type" field — thêm vào eval_questions.json nếu muốn phân nhóm
        boundary_type = item.get("boundary_type", "clear")

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(in_domain)}...")

        t0 = time.perf_counter()
        out = run_pipeline(
            query=q, config=cfg,
            qdrant_store=qdrant_store, bm25_data=bm25_data, embedder=embedder,
        )
        latency = time.perf_counter() - t0

        rejected = is_rejected(out.answer)
        if rejected:
            print(f"  ⚠️  FRR case [{tag}]: {q[:60]}...")

        results.append({
            "question": q,
            "tag": tag,
            "boundary_type": boundary_type,
            "rejected": rejected,
            "answer_preview": out.answer[:200],
            "latency": round(latency, 3),
        })

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FALSE REJECTION RATE")
    print("=" * 60)

    # Tổng thể
    total = len(results)
    rejected_total = sum(r["rejected"] for r in results)
    print(f"\n  Tổng in-domain: {total}")
    print(f"  Bị từ chối oan: {rejected_total}")
    print(f"  FRR tổng thể  : {rejected_total/total*100:.1f}%")

    # Phân theo boundary_type
    for btype in ["clear", "boundary"]:
        group = [r for r in results if r["boundary_type"] == btype]
        if not group:
            continue
        frr = sum(r["rejected"] for r in group) / len(group) * 100
        print(f"\n  FRR [{btype:8s}]: {sum(r['rejected'] for r in group)}/{len(group)} ({frr:.1f}%)")

    # Phân theo source_doc_tag
    print("\n  FRR theo source tag:")
    tags = sorted(set(r["tag"] for r in results))
    for tag in tags:
        g = [r for r in results if r["tag"] == tag]
        frr = sum(r["rejected"] for r in g) / len(g) * 100
        flag = " ⚠️" if frr > 5 else ""
        print(f"    {tag:30s}: {sum(r['rejected'] for r in g)}/{len(g)} ({frr:.1f}%){flag}")

    # Lưu kết quả
    out_path = BASE / "evaluation" / "results" / "frr_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu chi tiết → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(device=args.device)
