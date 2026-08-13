"""
rewrite.py — Viết lại câu hỏi y khoa bằng Ollama local (qwen3:4b).

Gọi Ollama qua requests.post() — KHÔNG dùng package `ollama` riêng.
Endpoint: http://localhost:11434/api/chat

Nguyên tắc:
  - Nếu LLM trả rỗng hoặc lỗi parse → fallback rewritten = original (log warning, không crash).
  - Nếu kết nối fail (ConnectionError/Timeout) → raise RuntimeError ngay.
  - KHÔNG gọi cloud API dù Ollama fail.
"""

from __future__ import annotations

import logging
import warnings

import requests

logger = logging.getLogger(__name__)

# ── System prompt CỐ ĐỊNH — không thay đổi ──────────────────────────────────
_REWRITE_PROMPT_TEMPLATE = """\
Bạn là công cụ viết lại câu hỏi y khoa. Nhiệm vụ: viết lại câu hỏi của bệnh nhân \
cho rõ ràng, đầy đủ ngữ cảnh, sửa lỗi chính tả nếu có — nhưng TUYỆT ĐỐI không thay đổi \
ý định gốc, không thêm giả định y khoa mà câu hỏi không có.
Chỉ trả về câu hỏi đã viết lại, không giải thích, không thêm ký tự nào khác.

Câu hỏi gốc: {query}
Câu hỏi viết lại:\
"""


def rewrite_query(
    query: str,
    model: str = "qwen3:4b",
    temperature: float = 0.0,
    ollama_url: str = "http://localhost:11434/api/chat",
    timeout: int = 30,
) -> tuple[str, str]:
    """Viết lại câu hỏi y khoa bằng Ollama local.

    Args:
        query:       Câu hỏi gốc từ người dùng.
        model:       Model Ollama dùng cho rewrite (mặc định qwen3:4b).
        temperature: Nhiệt độ sinh text (0.0 = deterministic).
        ollama_url:  URL Ollama chat API.
        timeout:     Timeout HTTP (giây).

    Returns:
        Tuple (original_query, rewritten_query).
        Nếu rewrite fail → rewritten_query = original_query (không crash).

    Raises:
        RuntimeError: Nếu không kết nối được Ollama.
    """
    prompt = _REWRITE_PROMPT_TEMPLATE.format(query=query.strip())

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": temperature},
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
        # HTTP lỗi (4xx/5xx) nhưng server đang chạy → fallback an toàn
        warnings.warn(
            f"[rewrite_query] HTTP error từ Ollama: {exc}. Dùng câu hỏi gốc.",
            stacklevel=2,
        )
        logger.warning("[rewrite_query] HTTP error: %s. Fallback to original.", exc)
        return query, query

    # Parse response
    try:
        data = response.json()
        rewritten = data["message"]["content"].strip()
    except (KeyError, ValueError) as exc:
        warnings.warn(
            f"[rewrite_query] Không parse được response Ollama: {exc}. Dùng câu hỏi gốc.",
            stacklevel=2,
        )
        logger.warning("[rewrite_query] Parse error: %s. Fallback to original.", exc)
        return query, query

    # Nếu LLM trả về rỗng hoặc chỉ whitespace → fallback
    if not rewritten:
        warnings.warn(
            "[rewrite_query] LLM trả về chuỗi rỗng. Dùng câu hỏi gốc.",
            stacklevel=2,
        )
        logger.warning("[rewrite_query] Empty response. Fallback to original.")
        return query, query

    logger.info("[rewrite_query] '%s' → '%s'", query, rewritten)
    return query, rewritten
