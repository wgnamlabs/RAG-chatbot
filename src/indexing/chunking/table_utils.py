"""
Tiện ích bảo vệ bảng Markdown khỏi bị chunker cắt đứt ngang.

Luồng sử dụng:
  1. Trước khi chunk: text_no_tables, tables = extract_tables(text)
  2. Đưa text_no_tables vào chunker.
  3. Với mỗi chunk đầu ra: pieces = restore_tables_as_chunks(chunk_text, tables)
     → mỗi bảng trở thành 1 phần tử riêng biệt (atomic), không lẫn ký tự khác.

QUAN TRỌNG — lý do chọn placeholder ASCII thuần:
  MarkdownHeaderTextSplitter và RecursiveCharacterTextSplitter xử lý nội bộ
  qua các bước normalize whitespace / strip control chars, do đó bất kỳ ký tự
  điều khiển nào (kể cả \\x00 NUL) đều bị xóa âm thầm → placeholder mất →
  bảng không bao giờ được khôi phục → dữ liệu bảng biến mất hoàn toàn.
  Chuỗi ASCII thuần như "TBLPLACEHOLDERSTART0000TBLPLACEHOLDEREND" không bao
  giờ xuất hiện trong Markdown y khoa, không bị bất kỳ splitter nào touch.
"""

import re
from typing import List, Tuple

# Placeholder thuần ASCII, không có ký tự điều khiển
_PLACEHOLDER_TMPL = "TBLPLACEHOLDERSTART{idx:04d}TBLPLACEHOLDEREND"
_PLACEHOLDER_RE   = re.compile(r"TBLPLACEHOLDERSTART(\d{4})TBLPLACEHOLDEREND")

# Khớp 1 bảng Markdown: 1 hoặc nhiều dòng liên tiếp bắt đầu bằng '|'
# Bắt cả dòng header + dòng separator (|---|) + các dòng data
_TABLE_BLOCK_RE = re.compile(
    r"(?:^\|[^\n]*\n)+",
    re.MULTILINE,
)

# Regex kiểm tra bảng rỗng: bảng chỉ có | và - không có chữ/số thật
_EMPTY_TABLE_RE = re.compile(r"^[|\-\s]+$")


def _is_empty_table(table_text: str) -> bool:
    """True nếu bảng không có nội dung chữ/số thật (chỉ khung | và -)."""
    stripped = re.sub(r"[|\-\s]", "", table_text)
    return len(stripped) == 0


def is_heading_only(text: str) -> bool:
    """True nếu chunk chỉ chứa breadcrumb và/hoặc dòng heading — không có
    nội dung thật.

    Dùng chung cho cả HierarchicalChunker và SemanticChunker để tránh
    copy-paste. Các chunk heading-only không có giá trị retrieval độc lập:
    thông tin đã nằm trong breadcrumb của chunk kế tiếp.

    Dòng "structural" bị bỏ qua:
      - Dòng heading Markdown: bắt đầu bằng '#'
      - Dòng breadcrumb:       bắt đầu bằng '[' và kết thúc bằng ']'
    """
    for line in text.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        # Có ít nhất 1 dòng nội dung thật → không phải heading-only
        return False
    return True


def extract_tables(text: str) -> Tuple[str, List[str]]:
    """Thay mỗi bảng Markdown bằng 1 placeholder ASCII duy nhất.

    Returns:
        text_no_tables: văn bản với bảng đã được thay bằng placeholder.
        tables: list các chuỗi bảng gốc (nguyên vẹn, giữ nguyên newline).
    """
    tables: List[str] = []

    def _replace(m: re.Match) -> str:
        idx = len(tables)
        tables.append(m.group(0).rstrip("\n"))
        return _PLACEHOLDER_TMPL.format(idx=idx) + "\n"

    new_text = _TABLE_BLOCK_RE.sub(_replace, text)
    return new_text, tables


def restore_tables_as_chunks(chunk_text: str, tables: List[str]) -> List[str]:
    """Tách placeholder trong chunk_text ra thành phần tử riêng (bảng atomic).

    Mỗi phần tử trả về là:
      - Đoạn văn bản thường (nếu có text trước/sau placeholder), hoặc
      - Chuỗi bảng nguyên vẹn — chỉ khi bảng có nội dung thật (không rỗng).
        Bảng chỉ có '|' và '-' (artifact PDF→Markdown) bị bỏ qua.

    Caller kiểm tra piece.strip().startswith("|") để đánh dấu is_table=True.

    Returns:
        List các chuỗi, luôn có ít nhất 1 phần tử.
    """
    parts: List[str] = []
    last_end = 0

    for m in _PLACEHOLDER_RE.finditer(chunk_text):
        pre = chunk_text[last_end : m.start()].strip()
        if pre:
            parts.append(pre)
        idx = int(m.group(1))
        table_text = tables[idx]
        # Bỏ qua bảng rỗng (artifact từ PDF→Markdown)
        if not _is_empty_table(table_text):
            parts.append(table_text)
        last_end = m.end()

    tail = chunk_text[last_end:].strip()
    if tail:
        parts.append(tail)

    if last_end == 0:
        return [chunk_text]
    return parts
