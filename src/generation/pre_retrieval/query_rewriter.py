"""
QueryRewriter — Viết lại câu hỏi để cải thiện retrieval, dùng qwen2.5:7b qua Ollama.

Tính năng:
  - Mở rộng viết tắt y khoa tiếng Việt (ĐTĐ → đái tháo đường, v.v.)
  - Sửa lỗi chính tả y khoa thường gặp
  - Thêm từ đồng nghĩa / thuật ngữ chuẩn
  - Few-shot prompt được thiết kế cho domain phụ sản
  - temperature=0.0 để output ổn định, deterministic
  - Fallback: nếu Ollama không chạy, trả về query gốc

Yêu cầu: Ollama đang chạy ở http://localhost:11434
         Model đã được pull: ollama pull qwen2.5:7b
"""

from dataclasses import dataclass, field
from typing import Optional
import re


# ── Từ điển viết tắt y khoa tiếng Việt ──────────────────────────────────────
MEDICAL_ABBREV_MAP = {
    r"\bĐTĐ\b":   "đái tháo đường",
    r"\bGDM\b":   "đái tháo đường thai kỳ",
    r"\bTSG\b":   "tiền sản giật",
    r"\bSG\b":    "sản giật",
    r"\bMLT\b":   "mổ lấy thai",
    r"\bBVPSHN\b": "bệnh viện phụ sản Hà Nội",
    r"\bDCTC\b":  "dụng cụ tử cung",
    r"\bKHHGĐ\b": "kế hoạch hóa gia đình",
    r"\bSKSS\b":  "sức khỏe sinh sản",
    r"\bHbA1c\b": "hemoglobin A1c",
    r"\bBMI\b":   "chỉ số khối cơ thể",
    r"\bHA\b":    "huyết áp",
    r"\bBN\b":    "bệnh nhân",
    r"\bPK\b":    "phòng khám",
    r"\bNK\b":    "nhiễm khuẩn",
    r"\bSA\b":    "siêu âm",
}


def _expand_abbreviations(text: str) -> str:
    """Mở rộng viết tắt y khoa trong query."""
    for pattern, expansion in MEDICAL_ABBREV_MAP.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


@dataclass
class QueryRewriterConfig:
    """Cấu hình cho QueryRewriter.

    Attributes:
        model:        Tên model Ollama (mặc định qwen2.5:7b).
        base_url:     URL Ollama server.
        temperature:  Luôn 0.0 để output deterministic.
        max_tokens:   Giới hạn độ dài output (rewrite ngắn, không cần dài).
        timeout:      Timeout HTTP request (giây).
        use_abbrev_expansion: Luôn mở rộng viết tắt trước khi gửi LLM.
        fallback_on_error: Nếu True, trả query gốc khi Ollama lỗi.
    """
    model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 150
    timeout: int = 30
    use_abbrev_expansion: bool = True
    fallback_on_error: bool = True


# ── Few-shot prompt ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """Bạn là chuyên gia cải thiện câu hỏi tìm kiếm cho tài liệu y khoa sản phụ khoa Việt Nam.

Nhiệm vụ: Viết lại câu hỏi để tìm kiếm hiệu quả hơn trong tài liệu y khoa, bằng cách:
1. Mở rộng viết tắt y khoa (ĐTĐ → đái tháo đường, v.v.)
2. Thêm thuật ngữ y khoa chuẩn
3. Giữ nguyên ý nghĩa gốc, không thêm thông tin mới
4. Trả lời CHỈ câu hỏi đã viết lại, không giải thích, không thêm gì khác
5. Nếu câu hỏi đã rõ ràng và chuẩn, giữ nguyên"""

_FEW_SHOT_EXAMPLES = [
    {
        "user": "ĐTĐ thai kỳ có nguy hiểm không?",
        "assistant": "đái tháo đường thai kỳ có nguy hiểm không?"
    },
    {
        "user": "Tiêu chuẩn chẩn đoán GDM bằng nghiệm pháp glucose là gì?",
        "assistant": "tiêu chuẩn chẩn đoán đái tháo đường thai kỳ bằng nghiệm pháp dung nạp glucose 75 gam là gì?"
    },
    {
        "user": "TSG và SG khác nhau thế nào?",
        "assistant": "tiền sản giật và sản giật khác nhau như thế nào về triệu chứng và xử trí?"
    },
    {
        "user": "Chế độ ăn cho bà bầu bị đường huyết cao?",
        "assistant": "chế độ dinh dưỡng và ăn uống cho thai phụ bị đái tháo đường thai kỳ tăng đường huyết?"
    },
]


class QueryRewriter:
    """Viết lại câu hỏi người dùng để cải thiện recall khi retrieval.

    Luồng:
      1. Mở rộng viết tắt y khoa (nếu use_abbrev_expansion=True)
      2. Gửi câu hỏi tới qwen2.5:7b qua Ollama để paraphrase
      3. Fallback về câu hỏi sau bước 1 nếu Ollama không phản hồi

    Ví dụ:
        rewriter = QueryRewriter()
        rewritten = rewriter.rewrite("ĐTĐ thai kỳ ảnh hưởng gì đến thai nhi?")
        # → "đái tháo đường thai kỳ ảnh hưởng như thế nào đến thai nhi và trẻ sơ sinh?"
    """

    def __init__(self, config: Optional[QueryRewriterConfig] = None):
        self.config = config or QueryRewriterConfig()

    def _build_messages(self, query: str) -> list:
        """Tạo danh sách messages theo format Ollama chat API."""
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for ex in _FEW_SHOT_EXAMPLES:
            messages.append({"role": "user",      "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": query})
        return messages

    def _call_ollama(self, query: str) -> Optional[str]:
        """Gọi Ollama API, trả về text viết lại hoặc None nếu lỗi."""
        try:
            import requests
            payload = {
                "model":   self.config.model,
                "messages": self._build_messages(query),
                "stream":  False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            }
            resp = requests.post(
                f"{self.config.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"[QueryRewriter] ⚠️  Ollama lỗi: {e}")
            return None

    def rewrite(self, query: str) -> str:
        """Viết lại câu hỏi. Trả về câu hỏi đã viết lại.

        Args:
            query: Câu hỏi gốc từ người dùng.

        Returns:
            Câu hỏi đã viết lại (hoặc query gốc nếu fallback).
        """
        # Bước 1: mở rộng viết tắt (rule-based, luôn áp dụng)
        expanded = _expand_abbreviations(query) if self.config.use_abbrev_expansion else query

        # Bước 2: LLM paraphrase
        rewritten = self._call_ollama(expanded)

        if rewritten:
            # Làm sạch output: bỏ dấu nháy, chỉ lấy dòng đầu tiên
            rewritten = rewritten.strip("'\"").split("\n")[0].strip()
            if rewritten:
                return rewritten

        # Fallback
        if self.config.fallback_on_error:
            return expanded  # ít nhất đã mở rộng viết tắt
        return query
