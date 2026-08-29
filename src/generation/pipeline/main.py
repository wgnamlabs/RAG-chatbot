"""
main.py — run_pipeline(): ghép toàn bộ generation pipeline.

Thứ tự bắt buộc:
    rewrite_query
    → hybrid_search (original + rewritten)
    → rerank (top_k=10)
    → dedup_redundant (final_top_k=5)
    → sandwich_order
    → build_prompt
    → Ollama generate (qwen3.5:4b, temperature=0.0, stream=False)
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
import re

import requests

from generation.schemas import Chunk, PipelineOutput
from generation.pipeline.rewrite import rewrite_query, expand_colloquial_terms
from generation.pipeline.retrieval import hybrid_search
from generation.pipeline.rerank import rerank
from generation.pipeline.postprocess import dedup_redundant, sandwich_order
from generation.pipeline.prompt import build_prompt
from generation.pipeline.safety_guard import apply_safety_guard

logger = logging.getLogger(__name__)

# ── Cấu hình mặc định ────────────────────────────────────────────────────────
DEFAULT_CONFIG: dict = {
    # Dùng qwen3.5:4b thay vì qwen3:4b — qwen3:4b có bug đã xác nhận
    # (Ollama issue #12234): cờ "think": false bị lờ hoàn toàn, khiến mọi
    # request đều tốn 500+ token suy luận ẩn dù không cần → latency 30-90s
    # cho tác vụ đơn giản. qwen3.5:4b tôn trọng đúng "think": false (đã
    # verify thực tế: eval_count 13, 0.19s cho câu chào, so với 537 token/
    # 8.5s khi không set think). Verify lại nếu đổi sang bản Qwen khác.
    "rewrite_model":    "qwen3.5:4b",  # Dùng chung model 4b cho nhẹ
    "generate_model":   "qwen3.5:4b",
    "top_k_retrieval":  15,
    "top_k_rerank":     10,    # rerank lấy top 10 để dedup có pool đủ lớn
    "final_top_k":      5,     # dedup cắt còn 5 trước khi vào prompt
    "rrf_k":            60,
    "sim_threshold":    0.9,
    "temperature":      0.0,
    "ollama_url":       "http://localhost:11434/api/chat",
    # ── OOD threshold — chiến lược "vùng xám 3 vùng" ─────────────────
    # Kết quả thực nghiệm (ood_score_analysis.py, 2026-08-22):
    #   OOD  : min=0.185, p10=0.273, median=0.358, p90=0.447, max=0.502
    #   InDom: min=0.379, p10=0.441, median=0.531, p90=0.621, max=0.720
    #   Overlap gap = +0.006 → KHÔNG có ngưỡng đơn nào đạt ≥90% OOD refusal
    #   và ≤5% FRR đồng thời (đã chứng minh toán học).
    #
    # Chiến lược 3 vùng:
    #   score < ood_threshold_hard (~0.27, OOD p10) → chắc chắn OOD → từ chối cứng.
    #   score ∈ [0.27, 0.45]                         → vùng xám → KHÔNG từ chối
    #     (retrieval + generation tự xử lý, tránh FRR trên câu hỏi triệu chứng
    #     thật có ngôn ngữ dân dã/chuyên sâu kỹ thuật có cosine sim tự nhiên thấp).
    #   score > 0.45                                  → in-domain rõ → không gate.
    #
    # Hệ quả: OOD refusal rate giảm (chỉ chặn ~57% OOD thay vì 83%), nhưng
    # loại hẳn FRR false-positive trên câu hỏi boundary và triệu chứng thật.
    # Đổi lại, vùng xám cần generation tự từ chối qua _NO_CONTEXT_ANSWER khi
    # không tìm được chunk liên quan — cơ chế này đã có sẵn.
    "ood_threshold_hard":  0.27,   # Chỉ từ chối cứng khi gần chắc chắn OOD (OOD p10)
    # Vùng xám [0.27, ood_gray_zone_upper]: generate bình thường, sau đó
    # gọi LLM tự đánh giá "câu hỏi có thuộc phạm vi 3 tài liệu này không?"
    # Nếu LLM nói không → truyền _NO_CONTEXT_ANSWER.
    # 0.0 = tắt gray-zone check (mặc định, để không tăng latency cho mọi request).
    "ood_gray_zone_upper": 0.45,   # Trên mức này = in-domain rõ, không cần kiểm tra
    "ood_llm_check_model": "qwen3.5:4b",  # Model nhỏ để check nhanh (~1-2s)
    # ── Rerank (DEPRECATED) ───────────────────────────────────────────
    # Eval 2024-08: rerank làm giảm MRR (0.9028 vs 0.9167), giảm P@5 (0.7111 vs 0.7556),
    # tăng latency P95 từ 0.14s lên 3.45s (~25×), do mismatch văn phong protocol lâm sàng.
    # Giữ code path để dễ thử reranker khác (ví dụ multilingual-e5) sau này.
    # Để bật lại: set use_rerank=True trong config.
    "use_rerank":       False,
    "rerank_model":     "BAAI/bge-reranker-v2-m3",
    "rerank_device":    "cpu",  # đổi sang "cuda" nếu có GPU
    "timeout_rewrite":  60,     # Tăng lên 60s vì model reasoning suy nghĩ rất lâu
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
        data = response.json()
        # ── DEBUG TẠM THỜI — xoá sau khi xác định xong nguyên nhân chậm ──────
        _thinking = (data.get("message", {}) or {}).get("thinking") or ""
        print(
            f"[DEBUG generate] eval_count={data.get('eval_count')} "
            f"eval_duration={data.get('eval_duration', 0)/1e9:.2f}s "
            f"load_duration={data.get('load_duration', 0)/1e9:.2f}s "
            f"prompt_eval_count={data.get('prompt_eval_count')} "
            f"thinking_len={len(_thinking)} "
            f"content_has_think_tag={'</think>' in data.get('message', {}).get('content', '')}"
        )
        # ── HẾT DEBUG ──────────────────────────────────────────────────────
        answer = data["message"]["content"].strip()
        # Xóa phần <think> hiệu quả hơn, đề phòng model không in thẻ mở <think>
        if '</think>' in answer:
            answer = answer.split('</think>')[-1].strip()
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"Không parse được response Ollama: {exc}") from exc

    return answer


# ── OOD LLM self-check (Phương án A) ────────────────────────────────────────
_OOD_CHECK_PROMPT = (
    "Bạn là bộ lọc nội dung cho chatbot tư vấn sức khỏe sản phụ khoa. "
    "Cần xác định câu hỏi này có thuộc phạm vi tư vấn của hệ thống không. "
    "Hệ thống chỉ trả lời câu hỏi thuộc 3 nhóm sau:\n"
    "1. Đái tháo đường thai kỳ (chẩn đoán, điều trị, insulin, theo dõi)\n"
    "2. Sản phụ khoa tổng quát (chăm sóc thai sản, sinh đẻ, tai biến sản khoa, "
    "biện pháp tránh thai)\n"
    "3. Sức khỏe sinh sản Việt Nam (vô sinh, xử lý thai ngoài tử cung, chăm sóc "
    "sơ sinh, dịch vụ SKSS)\n\n"
    "Câu hỏi: {question}\n\n"
    "Chỉ trả lời một trong hai: \"TRONG_PHAM_VI\" hoặc \"NGOAI_PHAM_VI\".\n"
    "Không giải thích, không viết thêm gì khác."
)


def _ood_llm_check(
    question: str,
    model: str,
    ollama_url: str,
    timeout: int = 20,
) -> bool:
    """Kiểm tra OOD bằng LLM judge cho vùng xám [ood_threshold_hard, ood_gray_zone_upper].

    Trả về True nếu LLM cho là TRONG_PHAM_VI (hoặc không phân biệt được).

    Fail-safe: nếu LLM không trả lời được (timeout, parse lỗi) → giả sử in-domain
    (trả về True) — thiếu lạc quan hơn là chặn oan câu hỏi khẩn cấp.
    """
    prompt = _OOD_CHECK_PROMPT.format(question=question.strip())
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.0},
        "think": False,
        "stream": False,
    }
    try:
        resp = requests.post(ollama_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        verdict = resp.json()["message"]["content"].strip().upper()
        # Xử lý các dạng trả lời: "TRONG_PHAM_VI", "NGOAI_PHAM_VI", hay lẫn nội dung khác
        if "NGOAI_PHAM_VI" in verdict and "TRONG_PHAM_VI" not in verdict:
            logger.info("[ood_llm_check] '%s...' → NGOAI_PHAM_VI", question[:50])
            return False
        return True   # mặc định: trong phạm vi
    except Exception as exc:
        logger.warning("[ood_llm_check] Fail (%s) → giả định in-domain.", exc)
        return True   # fail-safe: không chặn


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

    # Mở rộng thuật ngữ dân dã → y khoa (deterministic, áp dụng bất kể rewrite
    # qua LLM thành công hay rơi vào nhánh fallback ở trên) — chạy tại ĐÂY,
    # một điểm duy nhất, để đảm bảo mọi nhánh phía trên đều được phủ, không
    # phụ thuộc việc rewrite_query() có tự chuẩn hoá thuật ngữ hay không.
    rewritten_query = expand_colloquial_terms(rewritten_query)

    # ── Bước 2: hybrid_search ────────────────────────────────────────────────
    t0 = time.perf_counter()
    top1_dense_score: float = 0.0
    try:
        retrieved_chunks, top1_dense_score = hybrid_search(
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
            answer=apply_safety_guard(original_query, _NO_CONTEXT_ANSWER),
            sources_used=[],
            latency_breakdown={**latency, "rerank": 0.0, "generation": 0.0},
        )

    # OOD hard gate — chỉ từ chối khi score < ood_threshold_hard (~0.27 = OOD p10)
    # Tức gần chắc chắn OOD. Vùng xám [0.27, 0.45] được để retrieval +
    # generation tự xử lý — tránh FRR false-positive trên câu hỏi triệu chứng
    # thật có cosine sim thấp do ngôn ngữ dân dã hoặc thuật ngữ chuyên sâu.
    # Xem README mục OOD threshold để hiểu chiến lược 3 vùng.
    ood_hard: float = cfg.get("ood_threshold_hard", 0.0)
    if ood_hard > 0.0 and top1_dense_score < ood_hard:
        logger.info(
            "[run_pipeline] OOD hard gate: top1_dense=%.4f < hard=%.4f — từ chối.",
            top1_dense_score, ood_hard,
        )
        return PipelineOutput(
            original_query=original_query,
            rewritten_query=rewritten_query,
            answer=apply_safety_guard(original_query, _NO_CONTEXT_ANSWER),
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
            answer=apply_safety_guard(original_query, _ERROR_ANSWER),
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

    # Lớp an toàn cuối cùng — chạy SAU generate, dựa trên original_query
    # (không phụ thuộc retrieval/generation có thành công hay không), phát
    # hiện dấu hiệu nguy hiểm và:
    #   - Ghi đè answer nếu phát hiện bug tự-mâu-thuẫn (vừa từ chối vừa suy đoán)
    #   - Chèn thêm khuyến cáo khẩn cấp nếu answer chưa có ngôn ngữ dứt khoát
    #   - Giữ nguyên nếu câu hỏi không có dấu hiệu nguy hiểm, hoặc answer đã ổn
    answer = apply_safety_guard(original_query, answer)

    # ── Phương án A: OOD LLM check cho vùng xám ─────────────────────────────
    # Chỉ gọi khi score nằm trong [ood_threshold_hard, ood_gray_zone_upper].
    # Ngoài vùng này: hard gate đã xử lý (score thấp) hoặc in-domain rõ (score cao).
    ood_hard_cfg: float   = cfg.get("ood_threshold_hard",  0.0)
    ood_upper_cfg: float  = cfg.get("ood_gray_zone_upper", 0.0)
    in_gray_zone = (
        ood_hard_cfg > 0.0 and ood_upper_cfg > 0.0
        and ood_hard_cfg <= top1_dense_score < ood_upper_cfg
    )
    if in_gray_zone:
        t0 = time.perf_counter()
        is_in_scope = _ood_llm_check(
            question=original_query,
            model=cfg.get("ood_llm_check_model", cfg["rewrite_model"]),
            ollama_url=cfg["ollama_url"],
        )
        latency["ood_llm_check"] = time.perf_counter() - t0
        if not is_in_scope:
            # Nếu safety_guard đã chèn cảnh báo khẩn cấp vào answer (ví dụ dấu hiệu
            # nguy hiểm trong câu hỏi dân dã), GIỮ NGUYÊN — không mất cảnh báo an toàn.
            # Chỉ thay thế khi answer thuần túy (không có signal khẩn cấp).
            from generation.pipeline.safety_guard import detect_danger_signal
            if detect_danger_signal(original_query) is None:
                logger.info(
                    "[run_pipeline] OOD gray-zone check: NGOAI_PHAM_VI "
                    "(dense=%.4f, gray=[%.2f,%.2f]) — thay answer bằng _NO_CONTEXT_ANSWER.",
                    top1_dense_score, ood_hard_cfg, ood_upper_cfg,
                )
                answer = apply_safety_guard(original_query, _NO_CONTEXT_ANSWER)

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