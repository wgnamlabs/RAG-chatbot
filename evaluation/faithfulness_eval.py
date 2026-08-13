"""
faithfulness_eval.py — Đánh giá độ trung thực câu trả lời bằng qwen3:8b làm judge.

Chạy:
    python evaluation/faithfulness_eval.py --n 50
    python evaluation/faithfulness_eval.py --n 100 --save-cases

Output:
    - % faithful / partial / hallucinated in console
    - evaluation/results/faithfulness_results.json (chi tiết từng câu)
    - evaluation/results/hallucinated_cases.json (để bác sĩ review)

Lưu ý: Cần Ollama đang chạy với qwen3:8b và qwen3:4b.
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

import requests

from indexing.embedding import SentenceTransformerEmbedder
from indexing.embedding.config import EmbedderConfig
from indexing.vector_store import QdrantVectorStore, QdrantStoreConfig
from generation.pipeline.main import run_pipeline, DEFAULT_CONFIG
from generation.pipeline.prompt import _format_chunk

# ── Judge prompt ──────────────────────────────────────────────────────────────
_JUDGE_PROMPT = """\
Bạn là giám khảo đánh giá độ trung thực của câu trả lời y tế.

CONTEXT:
{context}

CÂU HỎI: {question}
CÂU TRẢ LỜI: {answer}

Chấm câu trả lời theo 3 mức, CHỈ dựa trên context trên, KHÔNG dùng kiến thức ngoài:
- "faithful": mọi thông tin y khoa trong câu trả lời đều có trong context
- "partial": có ý đúng nhưng thêm chi tiết không có trong context
- "hallucinated": có thông tin y khoa sai lệch hoặc bịa đặt so với context

Trả lời CHỈ bằng JSON hợp lệ (không có text thêm):
{{"label": "faithful|partial|hallucinated", "evidence": "trích câu trong context hỗ trợ hoặc mâu thuẫn", "reasoning": "giải thích ngắn"}}\
"""

def call_judge(question: str, context: str, answer: str,
               model: str = "qwen3:8b",
               ollama_url: str = "http://localhost:11434/api/chat") -> dict | None:
    """Gọi LLM judge, trả về dict hoặc None nếu parse lỗi."""
    prompt = _JUDGE_PROMPT.format(
        context=context[:4000],  # giới hạn context tránh vượt context window
        question=question,
        answer=answer,
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.0},
        "stream": False,
    }
    try:
        resp = requests.post(ollama_url, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
        # Tìm JSON trong response (LLM đôi khi thêm text trước/sau)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
        logging.warning("[judge] Parse lỗi: %s", exc)
    return None


def main(n_samples: int = 50, device: str = "cuda", save_cases: bool = False,
         ollama_url: str = "http://localhost:11434/api/chat") -> None:

    # ── Load eval set ─────────────────────────────────────────────────────────
    with open(BASE / "evaluation" / "eval_questions.json", encoding="utf-8") as f:
        qa_data = json.load(f)
    in_domain = [x for x in qa_data if x.get("ground_truth_sources")][:n_samples]
    print(f"Sẽ eval {len(in_domain)} câu in-domain (faithfulness judge)\n")

    # ── Load infra ────────────────────────────────────────────────────────────
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

    # ── Chạy pipeline + judge ─────────────────────────────────────────────────
    results = []
    label_counts = {"faithful": 0, "partial": 0, "hallucinated": 0, "error": 0}

    for i, item in enumerate(in_domain):
        q = item["question"]
        print(f"[{i+1}/{len(in_domain)}] {q[:60]}...", end=" ", flush=True)

        t0 = time.perf_counter()
        pipeline_out = run_pipeline(
            query=q, config=cfg,
            qdrant_store=qdrant_store, bm25_data=bm25_data, embedder=embedder,
        )
        latency = time.perf_counter() - t0

        answer = pipeline_out.answer
        if "Không tìm thấy thông tin" in answer or "lỗi kỹ thuật" in answer:
            label_counts["error"] += 1
            print("→ SKIP (pipeline từ chối/lỗi)")
            continue

        # Build context string từ sources_used
        context = "\n\n".join(
            _format_chunk(j, c) for j, c in enumerate(pipeline_out.sources_used)
        )

        # Gọi judge
        verdict = call_judge(q, context, answer, model="qwen3:8b", ollama_url=ollama_url)
        if verdict is None:
            label_counts["error"] += 1
            print("→ ERROR (judge parse fail)")
            continue

        label = verdict.get("label", "error").lower()
        if label not in label_counts:
            label = "error"
        label_counts[label] = label_counts.get(label, 0) + 1

        record = {
            "question": q,
            "answer": answer,
            "label": label,
            "evidence": verdict.get("evidence", ""),
            "reasoning": verdict.get("reasoning", ""),
            "latency_pipeline": round(latency, 3),
            "sources": [c.source for c in pipeline_out.sources_used],
        }
        results.append(record)
        print(f"→ {label.upper()}")

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    total_judged = sum(v for k, v in label_counts.items() if k != "error")
    print("\n" + "=" * 60)
    print("FAITHFULNESS RESULTS")
    print("=" * 60)
    for label in ["faithful", "partial", "hallucinated"]:
        count = label_counts[label]
        pct = count / total_judged * 100 if total_judged else 0
        print(f"  {label:15s}: {count:3d} ({pct:.1f}%)")
    print(f"  {'error/skip':15s}: {label_counts['error']:3d}")
    print(f"  {'TOTAL judged':15s}: {total_judged:3d}")

    # ── Lưu kết quả ───────────────────────────────────────────────────────────
    out_dir = BASE / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "faithfulness_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nĐã lưu chi tiết → {out_dir / 'faithfulness_results.json'}")

    if save_cases:
        bad = [r for r in results if r["label"] in ("partial", "hallucinated")]
        (out_dir / "hallucinated_cases.json").write_text(
            json.dumps(bad, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Đã lưu {len(bad)} case xấu → {out_dir / 'hallucinated_cases.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int, default=50)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--save-cases",  action="store_true",
                        help="Lưu hallucinated/partial cases riêng để bác sĩ review")
    parser.add_argument("--ollama-url",  default="http://localhost:11434/api/chat")
    args = parser.parse_args()
    main(n_samples=args.n, device=args.device, save_cases=args.save_cases,
         ollama_url=args.ollama_url)
