"""Tiện ích bảo vệ và chia bảng Markdown an toàn cho RAG.

Mục tiêu:
1. Không để MarkdownHeaderTextSplitter / RecursiveCharacterTextSplitter cắt ngang bảng.
2. Giữ caption kiểu ``**Bảng 1. ...**`` đi cùng bảng.
3. Nếu bảng vượt token budget, chia theo DATA ROW thay vì cắt ký tự ngẫu nhiên;
   mỗi part đều lặp lại caption + header + separator.
4. Dùng placeholder ASCII thuần để tránh bị splitter xóa mất.
"""

import re
from typing import Callable, List, Optional, Tuple


_PLACEHOLDER_TMPL = "TBLPLACEHOLDERSTART{idx:04d}TBLPLACEHOLDEREND"
_PLACEHOLDER_RE = re.compile(r"TBLPLACEHOLDERSTART(\d{4})TBLPLACEHOLDEREND")

# Caption table thường gặp sau clean.
_CAPTION_RE = re.compile(
    r"^(?:\*\*[^\n]+\*\*|(?:Bảng|BẢNG)\s+[^\n]+)$",
    re.IGNORECASE,
)

_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


def _split_md_row(row: str) -> List[str]:
    """Tách row Markdown đơn giản; corpus clean không dùng pipe escape phức tạp."""
    row = row.strip()
    if not row.startswith("|"):
        return []
    return [c.strip() for c in re.split(r"(?<!\\)\|", row.strip("|"))]


def _is_separator_row(row: str) -> bool:
    cells = _split_md_row(row)
    return bool(cells) and all(_SEPARATOR_CELL_RE.fullmatch(c) for c in cells)


def _table_lines(text: str) -> Tuple[List[str], List[str]]:
    """Return (caption_lines, table_rows)."""
    lines = [line.rstrip() for line in text.strip().splitlines()]
    first_table = next((i for i, line in enumerate(lines) if line.lstrip().startswith("|")), None)
    if first_table is None:
        return [], []
    caption = [x for x in lines[:first_table] if x.strip()]
    rows = [x for x in lines[first_table:] if x.lstrip().startswith("|")]
    return caption, rows


def _is_empty_table(table_text: str) -> bool:
    _, rows = _table_lines(table_text)
    if not rows:
        return True
    content = re.sub(r"[|:\-\s]", "", "\n".join(rows))
    return len(content) == 0


def is_table_chunk(text: str) -> bool:
    """True nếu text chứa một Markdown table hợp lệ (có header + separator)."""
    _, rows = _table_lines(text)
    return len(rows) >= 2 and _is_separator_row(rows[1])


def table_caption(text: str) -> str:
    caption, _ = _table_lines(text)
    return " ".join(line.strip() for line in caption).strip()


def is_heading_only(text: str) -> bool:
    """True nếu chunk chỉ chứa breadcrumb và/hoặc Markdown heading."""
    for line in text.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        return False
    return True


def extract_tables(text: str) -> Tuple[str, List[str]]:
    """Thay mỗi bảng Markdown bằng một placeholder ASCII.

    Nếu ngay trước bảng là caption dạng ``**...**`` hoặc ``Bảng ...``, caption
    được lấy ra khỏi body và gắn trực tiếp vào table chunk để retrieval không mất
    tên/ý nghĩa của bảng.
    """
    lines = text.splitlines()
    output: List[str] = []
    tables: List[str] = []
    i = 0

    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            output.append(lines[i])
            i += 1
            continue

        # Chỉ coi là bảng khi có ít nhất header + separator hợp lệ.
        if i + 1 >= len(lines) or not _is_separator_row(lines[i + 1]):
            output.append(lines[i])
            i += 1
            continue

        j = i
        table_rows: List[str] = []
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            table_rows.append(lines[j].rstrip())
            j += 1

        caption: Optional[str] = None
        # Cho phép blank line giữa caption và table, nhưng không làm mất paragraph
        # spacing nếu dòng trước KHÔNG phải caption.
        blank_count = 0
        while output and output[-1] == "":
            output.pop()
            blank_count += 1

        if output and _CAPTION_RE.fullmatch(output[-1].strip()):
            caption = output.pop().strip()
        else:
            output.extend([""] * blank_count)

        table_text = "\n".join(table_rows)
        if caption:
            table_text = f"{caption}\n\n{table_text}"
            # Giữ boundary rõ ràng giữa paragraph trước và placeholder.
            if output and output[-1] != "":
                output.append("")

        idx = len(tables)
        tables.append(table_text)
        output.append(_PLACEHOLDER_TMPL.format(idx=idx))
        i = j

    return "\n".join(output), tables


def restore_tables_as_chunks(chunk_text: str, tables: List[str]) -> List[str]:
    """Khôi phục placeholder; mỗi bảng trở thành một piece riêng biệt."""
    parts: List[str] = []
    last_end = 0

    for m in _PLACEHOLDER_RE.finditer(chunk_text):
        pre = chunk_text[last_end:m.start()].strip("\n")
        if pre:
            parts.append(pre)

        idx = int(m.group(1))
        if idx >= len(tables):
            raise IndexError(f"Table placeholder index {idx} vượt quá {len(tables)} tables")

        table_text = tables[idx]
        if not _is_empty_table(table_text):
            parts.append(table_text)
        last_end = m.end()

    tail = chunk_text[last_end:].strip("\n")
    if tail:
        parts.append(tail)

    if last_end == 0:
        return [chunk_text]
    return parts


def split_table_by_rows(
    table_text: str,
    max_tokens: int,
    token_len: Callable[[str], int],
    overlap_rows: int = 0,
) -> List[str]:
    """Chia table quá dài theo data rows, luôn lặp header + separator.

    Không cắt ngang một row. Nếu riêng một row + header đã vượt budget thì row đó
    vẫn được giữ nguyên trong một part; caller sẽ thấy ``too_long=True`` để audit.
    """
    if not is_table_chunk(table_text):
        return [table_text]
    if token_len(table_text) <= max_tokens:
        return [table_text]

    caption_lines, rows = _table_lines(table_text)
    header = rows[0]
    separator = rows[1]
    data_rows = rows[2:]

    prefix_lines = caption_lines + ([""] if caption_lines else []) + [header, separator]

    def build(part_rows: List[str]) -> str:
        return "\n".join(prefix_lines + part_rows).strip()

    if not data_rows:
        return [table_text]

    parts: List[str] = []
    current: List[str] = []
    idx = 0

    while idx < len(data_rows):
        row = data_rows[idx]
        candidate = current + [row]

        if current and token_len(build(candidate)) > max_tokens:
            parts.append(build(current))
            # Luôn bỏ ít nhất 1 row cũ để không thể lặp vô hạn khi overlap lớn.
            carry_n = min(overlap_rows, max(0, len(current) - 1))
            carry = current[-carry_n:] if carry_n > 0 else []
            current = list(carry)
            # Không tăng idx: thử lại row hiện tại với part mới.
            continue

        current = candidate
        idx += 1

        # Một row riêng đã quá dài: flush ngay để tránh vòng lặp.
        if len(current) == 1 and token_len(build(current)) > max_tokens:
            parts.append(build(current))
            current = []

    if current:
        parts.append(build(current))

    return parts or [table_text]
