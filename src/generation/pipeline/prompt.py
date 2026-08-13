"""
prompt.py — Build prompt đầy đủ từ chunks + câu hỏi.

Format mỗi chunk dùng section (breadcrumb) thay cho số trang,
vì tài liệu là Markdown (.md), không có số trang.

System prompt load từ file system_prompt.txt (cùng thư mục),
KHÔNG hardcode text trong Python để dễ chỉnh sửa nội dung prompt
mà không cần sửa code.
"""

from __future__ import annotations

import logging
from pathlib import Path

from generation.schemas import Chunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"
_SYSTEM_PROMPT_CACHE: str | None = None


def _load_system_prompt() -> str:
    """Load system prompt từ file, cache sau lần đầu."""
    global _SYSTEM_PROMPT_CACHE
    if _SYSTEM_PROMPT_CACHE is not None:
        return _SYSTEM_PROMPT_CACHE

    if not _SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy system_prompt.txt tại {_SYSTEM_PROMPT_PATH}. "
            "Hãy chắc chắn file tồn tại trong cùng thư mục với prompt.py."
        )

    _SYSTEM_PROMPT_CACHE = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    logger.debug("[prompt] Đã load system_prompt.txt (%d chars)", len(_SYSTEM_PROMPT_CACHE))
    return _SYSTEM_PROMPT_CACHE


def _format_chunk(i: int, chunk: Chunk) -> str:
    """Format 1 chunk thành chuỗi để đưa vào context block.

    Format: --- Nguồn {i+1}: {source} § {section} ---
            {text}
    """
    section_str = chunk.section or "N/A"
    header = f"--- Nguồn {i + 1}: {chunk.source} § {section_str} ---"
    return f"{header}\n{chunk.text}"


def build_prompt(user_question: str, chunks: list[Chunk]) -> str:
    """Tạo prompt đầy đủ để gửi cho LLM.

    Args:
        user_question: Câu hỏi của bệnh nhân (nên dùng rewritten_query).
        chunks:        Danh sách Chunk sau sandwich_order (đã sort + reorder).

    Returns:
        Chuỗi prompt hoàn chỉnh, sẵn sàng gửi cho Ollama.

    Raises:
        FileNotFoundError: Nếu system_prompt.txt không tồn tại.
    """
    # Format từng chunk, ghép cách nhau bằng 1 dòng trống
    formatted_chunks = [_format_chunk(i, c) for i, c in enumerate(chunks)]
    context_block = "\n\n".join(formatted_chunks)

    system_template = _load_system_prompt()
    prompt = system_template.format(
        context_block=context_block,
        user_question=user_question.strip(),
    )

    logger.debug(
        "[prompt] Built prompt: %d chunks, %d chars total",
        len(chunks), len(prompt),
    )
    return prompt
