"""
guardrails.py — Guardrails cho RAG chatbot y khoa phụ sản.

1. INPUT GUARDRAIL (trước khi retrieve):
   - Phát hiện câu hỏi yêu cầu chẩn đoán cá nhân cụ thể
   - Phát hiện câu hỏi hoàn toàn ngoài domain y tế phụ sản
   - Trả về cảnh báo rõ ràng thay vì câu trả lời y khoa

2. OUTPUT GUARDRAIL (sau khi generate):
   - Faithfulness check: kiểm tra câu trả lời có bịa thông tin không
   - Dùng qwen2.5:7b làm judge (cùng model, không cần load thêm)
   - Nếu phát hiện hallucination → thêm cảnh báo hoặc từ chối trả về
"""

import re
from typing import List, Optional


# ── Input Guardrail ────────────────────────────────────────────────────────────

# Pattern: câu hỏi chẩn đoán cá nhân (người dùng hỏi về bản thân họ)
_PERSONAL_DIAGNOSIS_PATTERNS = [
    r"tôi\s+bị",
    r"tôi\s+đang\s+bị",
    r"tôi\s+có\s+bị",
    r"vợ\s+tôi\s+bị",
    r"con\s+tôi\s+bị",
    r"em\s+bị",
    r"em\s+đang",
    r"em\s+có\s+bị",
    r"nên\s+uống\s+thuốc\s+gì",
    r"uống\s+thuốc\s+gì",
    r"dùng\s+thuốc\s+gì",
    r"có\s+nên\s+mổ\s+không",
    r"có\s+cần\s+mổ\s+không",
    r"kết\s+quả\s+xét\s+nghiệm",
    r"chỉ\s+số\s+của\s+tôi",
]

# Pattern: câu hỏi hoàn toàn ngoài domain y tế
_OUT_OF_DOMAIN_PATTERNS = [
    r"\bpython\b.*\blập\s+trình\b",
    r"\blập\s+trình\b",
    r"\btỷ\s+giá\b",
    r"\bchứng\s+khoán\b",
    r"\bxe\s+máy\b.*\bbảo\s+dưỡng\b",
    r"\bnhớt\b",
    r"\bbằng\s+lái\s+xe\b",
    r"\bluật\s+giao\s+thông\b",
    r"\btin\s+tức\b.*\bchính\s+trị\b",
]

_PERSONAL_DIAGNOSIS_RESPONSE = (
    "Xin lỗi, tôi không thể đưa ra chẩn đoán hoặc lời khuyên điều trị cá nhân.\n\n"
    "Chatbot này chỉ cung cấp thông tin tham khảo từ tài liệu y khoa phụ sản. "
    "Để được tư vấn cụ thể về tình trạng sức khỏe của bạn, vui lòng:\n"
    "- Đến khám trực tiếp tại cơ sở y tế\n"
    "- Liên hệ bác sĩ hoặc nhân viên y tế có chuyên môn\n\n"
    "Tôi có thể giúp bạn tìm hiểu thông tin chung về các chủ đề sức khỏe phụ sản."
)

_OUT_OF_DOMAIN_RESPONSE = (
    "Câu hỏi này nằm ngoài phạm vi của chatbot tư vấn sức khỏe phụ sản.\n\n"
    "Tôi chỉ có thể trả lời các câu hỏi liên quan đến:\n"
    "- Đái tháo đường thai kỳ\n"
    "- Chăm sóc sức khỏe sinh sản\n"
    "- Thực hành lâm sàng sản phụ khoa\n\n"
    "Vui lòng đặt câu hỏi trong phạm vi trên."
)


class InputGuardrail:
    """Kiểm tra câu hỏi đầu vào trước khi đưa vào pipeline RAG."""

    def check(self, query: str) -> Optional[str]:
        """Kiểm tra query. Trả về None nếu hợp lệ, hoặc response từ chối nếu vi phạm.

        Args:
            query: Câu hỏi của người dùng.

        Returns:
            None nếu query hợp lệ (tiếp tục pipeline).
            String response nếu query vi phạm (trả về ngay, bỏ qua pipeline).
        """
        q = query.lower()

        # Kiểm tra ngoài domain trước (ưu tiên cao nhất)
        for pattern in _OUT_OF_DOMAIN_PATTERNS:
            if re.search(pattern, q):
                return _OUT_OF_DOMAIN_RESPONSE

        # Kiểm tra chẩn đoán cá nhân
        for pattern in _PERSONAL_DIAGNOSIS_PATTERNS:
            if re.search(pattern, q):
                return _PERSONAL_DIAGNOSIS_RESPONSE

        return None  # Query hợp lệ


# ── Output Guardrail ───────────────────────────────────────────────────────────

_FAITHFULNESS_SYSTEM = """Bạn là kiểm duyệt viên y tế. Nhiệm vụ: kiểm tra câu trả lời có bịa thông tin không.

Chỉ trả lời PASS hoặc FAIL:
- PASS: mọi thông tin trong câu trả lời đều có cơ sở trong ngữ cảnh
- FAIL: câu trả lời chứa thông tin KHÔNG có trong ngữ cảnh (hallucination)"""

_FAITHFULNESS_WARNING = (
    "\n\n---\n"
    "⚠️ **Cảnh báo hệ thống**: Câu trả lời này có thể chứa thông tin không có "
    "trong tài liệu gốc. Vui lòng kiểm tra lại với bác sĩ hoặc tài liệu y khoa chính thức."
)


class OutputGuardrail:
    """Kiểm tra câu trả lời sau khi generate — Faithfulness check.

    Dùng qwen2.5:7b (cùng model generation) làm LLM judge.
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: OllamaLLMClient đã khởi tạo. Nếu None, tự tạo.
        """
        self._client = llm_client

    def _get_client(self):
        if self._client is None:
            from .llm_client import OllamaLLMClient, LLMConfig
            self._client = OllamaLLMClient(LLMConfig(
                model="qwen2.5:7b",
                temperature=0.0,
                max_tokens=10,  # Chỉ cần PASS/FAIL
            ))
        return self._client

    def check_faithfulness(
        self,
        answer: str,
        contexts: List[dict],
        add_warning_on_fail: bool = True,
    ) -> tuple[str, bool]:
        """Kiểm tra faithfulness của câu trả lời với context.

        Args:
            answer:              Câu trả lời từ LLM.
            contexts:            Danh sách context dict (có key "text").
            add_warning_on_fail: Thêm cảnh báo vào answer nếu FAIL.

        Returns:
            (final_answer, is_faithful) — final_answer đã thêm warning nếu cần.
        """
        if not contexts or not answer.strip():
            return answer, True

        context_text = "\n\n".join(ctx.get("text", "") for ctx in contexts[:5])

        messages = [
            {"role": "system", "content": _FAITHFULNESS_SYSTEM},
            {"role": "user",   "content": (
                f"Ngữ cảnh:\n{context_text[:3000]}\n\n"
                f"Câu trả lời:\n{answer[:1000]}\n\n"
                "Trả lời: PASS hoặc FAIL"
            )},
        ]

        try:
            verdict = self._get_client().chat(messages).upper().strip()
            is_faithful = "PASS" in verdict

            if not is_faithful and add_warning_on_fail:
                answer = answer + _FAITHFULNESS_WARNING

            return answer, is_faithful
        except Exception as e:
            print(f"[OutputGuardrail] ⚠️  Faithfulness check lỗi: {e}")
            return answer, True  # Assume faithful nếu không check được
