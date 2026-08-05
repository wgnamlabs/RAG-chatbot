"""
prompt_builder.py — Xây dựng prompt RAG cho domain y khoa phụ sản.

Nguyên tắc thiết kế:
  1. Citation bắt buộc: mọi claim phải có [NGUỒN N] tham chiếu về context.
  2. "Không biết" rõ ràng: nếu context không đủ, LLM phải thừa nhận.
  3. Disclaimer y khoa: mọi câu trả lời về điều trị/liều lượng phải có cảnh báo.
  4. Không bịa đặt: LLM được chỉ dẫn rõ chỉ dùng thông tin từ context.
"""

from typing import List


# ── Disclaimer y khoa ─────────────────────────────────────────────────────────
MEDICAL_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Lưu ý quan trọng**: Thông tin trên chỉ mang tính tham khảo từ tài liệu y khoa. "
    "Không thay thế lời khuyên, chẩn đoán hoặc điều trị của bác sĩ. "
    "Mọi quyết định y tế cần được thực hiện dưới sự hướng dẫn của nhân viên y tế có chuyên môn."
)

# Keywords kích hoạt disclaimer
_TREATMENT_KEYWORDS = [
    "điều trị", "liều", "thuốc", "insulin", "metformin", "tiêm", "uống",
    "phẫu thuật", "mổ", "thủ thuật", "kháng sinh", "vaccine",
    "treatment", "dosage", "medication", "inject",
]


def _needs_disclaimer(answer: str) -> bool:
    """Kiểm tra xem câu trả lời có cần disclaimer y khoa không."""
    answer_lower = answer.lower()
    return any(kw in answer_lower for kw in _TREATMENT_KEYWORDS)


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """Bạn là trợ lý tư vấn sức khỏe phụ sản, cung cấp thông tin từ tài liệu y khoa chính thức.

**Nguyên tắc bắt buộc:**
1. Chỉ trả lời dựa trên NGỮCẢNH được cung cấp. Không thêm thông tin từ kiến thức ngoài.
2. Mỗi thông tin quan trọng PHẢI được trích dẫn theo định dạng [NGUỒN N] ngay sau câu đó.
3. Nếu ngữ cảnh không có thông tin cần thiết, trả lời: "Tôi không tìm thấy thông tin này trong tài liệu tham khảo."
4. Không được đoán mò, bịa đặt, hay suy luận ngoài phạm vi tài liệu.
5. Trả lời bằng tiếng Việt, ngắn gọn và chính xác."""


def build_prompt(
    query: str,
    contexts: List[dict],
    include_sources: bool = True,
) -> tuple[list, list]:
    """Tạo messages list cho Ollama chat API và danh sách sources.

    Args:
        query:           Câu hỏi của người dùng (đã qua rewrite nếu có).
        contexts:        Danh sách dict {"text": str, "metadata": dict, "score": float}.
                         Thứ tự trong list = thứ tự source [NGUỒN 1], [NGUỒN 2], ...
        include_sources: Thêm thông tin nguồn vào context block.

    Returns:
        (messages, sources_list) — messages theo format Ollama, sources_list là
        danh sách dict {index, source_file, chunk_id} để trả về cho UI.
    """
    if not contexts:
        # Không có context → trả lời từ chối
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": (
                f"Câu hỏi: {query}\n\n"
                "Ngữ cảnh: (Không tìm thấy tài liệu liên quan)\n\n"
                "Hãy trả lời câu hỏi."
            )},
        ]
        return messages, []

    # Xây dựng context block
    context_lines = []
    sources_list  = []

    for i, ctx in enumerate(contexts, start=1):
        text        = ctx.get("text", "")
        metadata    = ctx.get("metadata", {})
        source_file = metadata.get("source_file", metadata.get("source", "unknown"))
        chunk_id    = metadata.get("chunk_id", "")

        if include_sources:
            context_lines.append(f"[NGUỒN {i}] (Tệp: {source_file})\n{text}")
        else:
            context_lines.append(f"[NGUỒN {i}]\n{text}")

        sources_list.append({
            "index":       i,
            "source_file": source_file,
            "chunk_id":    chunk_id,
        })

    context_block = "\n\n".join(context_lines)

    user_content = (
        f"Ngữ cảnh tài liệu:\n\n{context_block}\n\n"
        f"---\n\n"
        f"Câu hỏi: {query}\n\n"
        "Hãy trả lời câu hỏi dựa trên ngữ cảnh trên. "
        "Trích dẫn nguồn theo dạng [NGUỒN N] cho mỗi thông tin quan trọng."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return messages, sources_list


def add_disclaimer_if_needed(answer: str) -> str:
    """Thêm disclaimer y khoa nếu câu trả lời đề cập đến điều trị/thuốc."""
    if _needs_disclaimer(answer):
        return answer + MEDICAL_DISCLAIMER
    return answer
