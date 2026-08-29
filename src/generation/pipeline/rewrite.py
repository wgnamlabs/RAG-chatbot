"""
rewrite.py — Viết lại câu hỏi y khoa bằng Ollama local (qwen3.5:4b).

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
import re

import requests

logger = logging.getLogger(__name__)

# ── Từ điển đồng nghĩa dân dã → y khoa ──────────────────────────────────────
# Bối cảnh: eval thực tế cho thấy câu hỏi "thai máy yếu dần, không thấy máy
# nữa" bị retrieval miss vì tài liệu gốc chỉ dùng thuật ngữ "cử động thai",
# không có từ "thai máy" ở đâu cả. rewrite_query qua LLM (qwen3) KHÔNG đảm
# bảo tự chuẩn hoá đúng thuật ngữ mỗi lần (không deterministic), nên xử lý
# bằng một bước substitution cứng, chạy SAU LLM rewrite (hoặc sau fallback),
# để đảm bảo query gửi vào retrieval luôn có đủ token y khoa cần thiết cho
# cả BM25 (khớp từ) lẫn dense embedding (khớp ngữ nghĩa).
#
# Nguyên tắc: CHỈ NỐI THÊM thuật ngữ y khoa vào cuối câu (dạng chú thích),
# KHÔNG thay thế/xoá từ dân dã gốc — giữ đúng tinh thần "không thay đổi ý
# định gốc" của rewrite prompt phía trên.
COLLOQUIAL_MEDICAL_SYNONYMS: dict[str, str] = {
    "thai máy": "cử động thai",
    "không thấy máy": "giảm cử động thai",
    "không thấy đạp": "giảm cử động thai",
    "không đạp nữa": "giảm cử động thai",
    "bé không đạp": "giảm cử động thai",
    "thai không máy": "thai không cử động",
    "im re không đạp": "giảm cử động thai",
    "đau bụng đẻ": "cơn co tử cung chuyển dạ",
    "vỡ nước ối": "vỡ ối",
    "ra nước ối": "vỡ ối",
    "ra máu tươi": "xuất huyết âm đạo",
    "ra máu nhiều": "xuất huyết âm đạo",
    "băng huyết": "băng huyết sau sinh",
    "đau bụng dữ dội": "đau bụng cấp",
}


def expand_colloquial_terms(text: str) -> str:
    """
    Nối thêm thuật ngữ y khoa tương ứng (nếu tìm thấy cụm dân dã đã biết)
    vào cuối câu, dạng chú thích trong ngoặc. Không sửa/xoá gì trong câu gốc.

    Ví dụ:
        "Bụng em bé không thấy đạp từ sáng tới giờ, thai 36 tuần"
        → "Bụng em bé không thấy đạp từ sáng tới giờ, thai 36 tuần
           (thuật ngữ y khoa liên quan: giảm cử động thai)"
    """
    text_lower = text.lower()
    matched_terms: list[str] = []
    for colloquial, medical in COLLOQUIAL_MEDICAL_SYNONYMS.items():
        if colloquial in text_lower and medical.lower() not in text_lower:
            if medical not in matched_terms:
                matched_terms.append(medical)

    if not matched_terms:
        return text

    suffix = f" (thuật ngữ y khoa liên quan: {', '.join(matched_terms)})"
    logger.info("[rewrite_query] Mở rộng thuật ngữ dân dã: %s", matched_terms)
    return text.rstrip() + suffix

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
    model: str = "qwen3.5:4b",
    temperature: float = 0.0,
    ollama_url: str = "http://localhost:11434/api/chat",
    timeout: int = 30,
) -> tuple[str, str]:
    """Viết lại câu hỏi y khoa bằng Ollama local.

    Args:
        query:       Câu hỏi gốc từ người dùng.
        model:       Model Ollama dùng cho rewrite (mặc định qwen3.5:4b).
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

    # Yêu cầu Ollama >= v0.9.0 để tắt thinking mode
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": temperature},
        "think": False,
        "keep_alive": "30m",   # Giữ model trong VRAM 30 phút, tránh re-load
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
        # ── DEBUG TẠM THỜI — xoá sau khi xác định xong nguyên nhân chậm ──────
        _thinking = (data.get("message", {}) or {}).get("thinking") or ""
        print(
            f"[DEBUG rewrite] eval_count={data.get('eval_count')} "
            f"eval_duration={data.get('eval_duration', 0)/1e9:.2f}s "
            f"load_duration={data.get('load_duration', 0)/1e9:.2f}s "
            f"prompt_eval_count={data.get('prompt_eval_count')} "
            f"thinking_len={len(_thinking)} "
            f"content_has_think_tag={'</think>' in data.get('message', {}).get('content', '')}"
        )
        # ── HẾT DEBUG ──────────────────────────────────────────────────────
        rewritten = data["message"]["content"].strip()
        # Xóa phần suy nghĩ đề phòng thẻ mở bị nuốt
        if '</think>' in rewritten:
            rewritten = rewritten.split('</think>')[-1].strip()
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