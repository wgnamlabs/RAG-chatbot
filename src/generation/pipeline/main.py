"""
main.py — run_pipeline(): ghép toàn bộ generation pipeline.

Thứ tự bắt buộc:
    rewrite_query
    → hybrid_search (original + rewritten)
    → rerank (top_k=10)
    → dedup_redundant (final_top_k=5)
    → sandwich_order
    → build_prompt
    → Ollama generate (qwen3:8b, temperature=0.0, stream=False)
    → PipelineOutput

Nguyên tắc:
  - Đo latency từng bước bằng time.perf_counter() (KHÔNG dùng time.time()).
  - Bắt exception ở TỪNG bước — nếu lỗi, log rõ, trả về PipelineOutput lỗi graceful.
  - Nếu retrieval trả về [] → trả về "không tìm thấy" ngay, không gọi Ollama.
  - KHÔNG gọi cloud API dù Ollama fail — raise RuntimeError rõ ràng từ rewrite_query.
  - KHÔNG mock hay fallback sang cloud LLM.
"""

from __future__ import annotations

import logging
import time
import warnings

import requests

from generation.schemas import Chunk, PipelineOutput
from generation.pipeline.rewrite import rewrite_query
from generation.pipeline.retrieval import hybrid_search
from generation.pipeline.rerank import rerank
from generation.pipeline.postprocess import dedup_redundant, sandwich_order
from generation.pipeline.prompt import build_prompt

logger = logging.getLogger(__name__)

# ── Cấu hình mặc định ────────────────────────────────────────────────────────
DEFAULT_CONFIG: dict = {
    "rewrite_model":    "qwen3:8b",    # Dùng chung 1 model 8B cho cả rewrite và generate để tránh tràn VRAM
    "generate_model":   "qwen3:8b",
    "top_k_retrieval":  15,
    "top_k_rerank":     10,    # rerank lấy top 10 để dedup có pool đủ lớn
    "final_top_k":      5,     # dedup cắt còn 5 trước khi vào prompt
    "rrf_k":            60,
    "sim_threshold":    0.9,
    "temperature":      0.0,
    "ollama_url":       "http://localhost:11434/api/chat",
    # ── Rerank (DEPRECATED) ───────────────────────────────────────────
    # Eval 2024-08: rerank làm giảm MRR (0.9028 vs 0.9167), giảm P@5 (0.7111 vs 0.7556),
    # tăng latency P95 từ 0.14s lên 3.45s (~25×), do mismatch văn phong protocol lâm sàng.
    # Giữ code path để dễ thử reranker khác (ví dụ multilingual-e5) sau này.
    # Để bật lại: set use_rerank=True trong config.
    "use_rerank":       False,
    "rerank_model":     "BAAI/bge-reranker-v2-m3",
    "rerank_device":    "cpu",  # đổi sang "cuda" nếu có GPU
    "timeout_rewrite":  30,
    "timeout_generate": 120,
}

# Câu trả lời mặc định khi không tìm thấy context
_NO_CONTEXT_ANSWER = "Không tìm thấy thông tin liên quan trong tài liệu y tế sản phụ khoa."
_ERROR_ANSWER      = "Xin lỗi, hệ thống gặp lỗi kỹ thuật. Vui lòng thử lại sau."


def _call_ollama_generate(
    prompt: str,
    model: str,
    temperature: float,
    ollama_url: str,
    timeout: int,
) -> str:
    """Gọi Ollama generate (blocking, không stream).

    Returns:
        Chuỗi câu trả lời từ LLM.

    Raises:
        RuntimeError: Nếu không kết nối được Ollama.
    """
    # Yêu cầu Ollama >= v0.9.0 để nhận diện param "think": False ở top-level payload
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": temperature},
        "think": False,
        "keep_alive": "30m",
        "stream": False,
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=timeout)
        response.raise_for_status()
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise RuntimeError(
            f"Ollama không khả dụng tại localhost:11434. Chi tiết: {exc}"
        ) from exc
    except requests.HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP error: {exc}") from exc

    try:
        answer = response.json()["message"]["content"].strip()
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"Không parse được response Ollama: {exc}") from exc

    return answer


def run_pipeline(
    query: str,
    config: dict | None = None,
    qdrant_store=None,      # QdrantVectorStore đã load()
    bm25_data: dict = None,
    embedder=None,          # SentenceTransformerEmbedder đã load()
    cross_encoder=None,     # Không dùng trực tiếp — rerank() tự load nếu None
) -> PipelineOutput:
    """Chạy toàn bộ RAG pipeline từ câu hỏi đến câu trả lời.

    Args:
        query:        Câu hỏi thô từ người dùng.
        config:       Dict cấu hình (merge với DEFAULT_CONFIG).
        qdrant_store: QdrantVectorStore đã load().
        bm25_data:    Dict từ bm25_index.pkl.
        embedder:     SentenceTransformerEmbedder đã load().
        cross_encoder: Không dùng (placeholder cho tương lai cache ngoài).

    Returns:
        PipelineOutput với answer, sources_used, latency_breakdown.
        Luôn trả về PipelineOutput (không raise exception ra ngoài) —
        lỗi được bắt và trả về trong field `answer`.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    latency: dict[str, float] = {}
    original_query = query

    # ── Bước 1: rewrite_query ────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        original_query, rewritten_query = rewrite_query(
            query=query,
            model=cfg["rewrite_model"],
            temperature=cfg["temperature"],
            ollama_url=cfg["ollama_url"],
            timeout=cfg["timeout_rewrite"],
        )
    except RuntimeError as exc:
        # Ollama không khả dụng: fallback rewrite = original, tiếp tục pipeline
        warnings.warn(f"[run_pipeline] rewrite_query fail: {exc}. Dùng câu hỏi gốc.", stacklevel=2)
        logger.warning("[run_pipeline] rewrite_query fail: %s", exc)
        rewritten_query = query
    except Exception as exc:
        logger.error("[run_pipeline] rewrite_query lỗi bất ngờ: %s", exc)
        rewritten_query = query
    latency["rewrite"] = time.perf_counter() - t0

    # ── Bước 2: hybrid_search ────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        retrieved_chunks = hybrid_search(
            original_query=original_query,
            rewritten_query=rewritten_query,
            qdrant_store=qdrant_store,
            bm25_data=bm25_data,
            embedder=embedder,
            top_k=cfg["top_k_retrieval"],
            rrf_k=cfg["rrf_k"],
        )
    except Exception as exc:
        logger.error("[run_pipeline] hybrid_search lỗi: %s", exc)
        retrieved_chunks = []
    latency["retrieval"] = time.perf_counter() - t0

    # Không tìm thấy chunk nào → trả về gracefully, không gọi Ollama
    if not retrieved_chunks:
        logger.info("[run_pipeline] Retrieval trả về [] — từ chối trả lời.")
        return PipelineOutput(
            original_query=original_query,
            rewritten_query=rewritten_query,
            answer=_NO_CONTEXT_ANSWER,
            sources_used=[],
            latency_breakdown={**latency, "rerank": 0.0, "generation": 0.0},
        )

    # ── Bước 3: rerank (DEPRECATED — tắt theo mặc định) ──────────────────────
    t0 = time.perf_counter()
    if cfg.get("use_rerank", False):
        warnings.warn(
            "[run_pipeline] rerank đang bật (use_rerank=True). "
            "Eval 2024-08 cho thấy rerank giảm MRR/Precision và tăng latency ~25×. "
            "Chỉ bật nếu đã kiểm chứng reranker mới phù hợp domain.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            reranked_chunks = rerank(
                query=rewritten_query,
                chunks=retrieved_chunks,
                top_k=cfg["top_k_rerank"],
                model_name=cfg["rerank_model"],
                device=cfg["rerank_device"],
            )
        except Exception as exc:
            logger.error("[run_pipeline] rerank lỗi: %s. Dùng retrieved_chunks thô.", exc)
            reranked_chunks = retrieved_chunks[:cfg["top_k_rerank"]]
    else:
        # Hybrid RRF là default (không rerank) — cắt thẳng top_k_rerank chunk
        reranked_chunks = retrieved_chunks[:cfg["top_k_rerank"]]
    latency["rerank"] = time.perf_counter() - t0

    # ── Bước 4: dedup_redundant ──────────────────────────────────────────────
    try:
        deduped_chunks = dedup_redundant(
            chunks=reranked_chunks,
            embedder=embedder,
            sim_threshold=cfg["sim_threshold"],
            final_top_k=cfg["final_top_k"],
        )
    except Exception as exc:
        logger.error("[run_pipeline] dedup_redundant lỗi: %s. Dùng reranked_chunks.", exc)
        deduped_chunks = reranked_chunks[:cfg["final_top_k"]]

    # ── Bước 5: sandwich_order ───────────────────────────────────────────────
    try:
        final_chunks = sandwich_order(deduped_chunks)
    except Exception as exc:
        logger.error("[run_pipeline] sandwich_order lỗi: %s. Dùng deduped_chunks.", exc)
        final_chunks = deduped_chunks

    # ── Bước 6: build_prompt ─────────────────────────────────────────────────
    try:
        prompt_text = build_prompt(
            user_question=rewritten_query,
            chunks=final_chunks,
        )
    except Exception as exc:
        logger.error("[run_pipeline] build_prompt lỗi: %s", exc)
        return PipelineOutput(
            original_query=original_query,
            rewritten_query=rewritten_query,
            answer=_ERROR_ANSWER,
            sources_used=final_chunks,
            latency_breakdown={**latency, "generation": 0.0},
        )

    # ── Bước 7: Ollama generate ──────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        answer = _call_ollama_generate(
            prompt=prompt_text,
            model=cfg["generate_model"],
            temperature=cfg["temperature"],
            ollama_url=cfg["ollama_url"],
            timeout=cfg["timeout_generate"],
        )
    except RuntimeError as exc:
        logger.error("[run_pipeline] Ollama generate fail: %s", exc)
        answer = _ERROR_ANSWER
    except Exception as exc:
        logger.error("[run_pipeline] generate lỗi bất ngờ: %s", exc)
        answer = _ERROR_ANSWER
    latency["generation"] = time.perf_counter() - t0

    latency["total"] = sum(latency.values())

    logger.info(
        "[run_pipeline] Hoàn thành | latency=%s | sources=%d",
        {k: f"{v:.2f}s" for k, v in latency.items()},
        len(final_chunks),
    )

    return PipelineOutput(
        original_query=original_query,
        rewritten_query=rewritten_query,
        answer=answer,
        sources_used=final_chunks,
        latency_breakdown=latency,
    )
