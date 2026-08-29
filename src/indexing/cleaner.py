"""
Làm sạch file Markdown xuất ra từ Docling, chuẩn bị cho bước chunking.

Nguyên tắc: Clean format mạnh — clean content yếu.
  - Được phép sửa: entity HTML, format bullet/heading, khoảng trắng, TOC rác.
  - Không được sửa: số, đơn vị, ngưỡng lâm sàng, liều thuốc, tuổi thai, %.
  - Với heading (ảnh hưởng trực tiếp đến chunking): CHỈ tự động sửa khi độ tin
    cậy cao; trường hợp mơ hồ → KHÔNG động vào nội dung, chỉ ghi warning vào
    report để review tay.

Các bước xử lý:
 0. Chuẩn hóa line ending: CRLF/CR → LF.
 1. NFC Unicode normalization.
 2. HTML entity decode  (&lt; → <, &amp; → &, &#160; → ' ', ...)
 3. Legacy font fix  (hiện để trống — 7 file mới không có lỗi VNI/TCVN3)
 4. PUA character fix  (Symbol/Wingdings) — chỉ chạy khi file chứa PUA.
 5. Remove TOC block  (## MỤC LỤC ... heading kế) — chỉ khi block có dot-leader thật.
 6. Repair broken headings — chỉ khi 2 dòng LIỀN NHAU, heading không kết thúc dấu câu.
 6a. Promote heading đánh số bị "đội lốt" bullet '- '  (Docling/PDF convert
     heading số La Mã hoặc heading N.N. thành bullet '- '):
     - '- I. Text' / '- II. Text' / '- C. Text' (chữ La Mã hoặc 1 chữ cái đơn,
       đứng riêng)  → tự thêm '## ' (HIGH confidence).
     - '- 1.2. Text' / '- 4.4.7. Text' (nhiều cấp số phân tách bằng dấu chấm)
       → tự thêm '## ' rồi để bước 7f chuẩn hoá cấp heading theo số lượng cấp
       số (HIGH confidence — mẫu số nhiều cấp là tín hiệu heading rất đặc thù,
       hiếm khi trùng với bullet nội dung thật trong bộ tài liệu này).
 6b. Promote numbered headings thiếu '#'  (heading bị Docling bỏ sót dấu #):
     - HIGH confidence (ALL CAPS, đứng riêng sau dòng trắng/câu hoàn chỉnh/bullet) → tự thêm '##'.
     - MEDIUM confidence (Title Case, ≤8 từ, đứng riêng sau dòng trắng) → tự thêm '###'.
     - LOW confidence (câu dài, mơ hồ) → KHÔNG sửa, chỉ ghi warning để review tay.
 6c. Repair heading bị vỡ nối với đoạn văn dài phía sau (heading kết thúc bằng
     dấu '-' hoặc giới từ/liên từ lửng lơ, đoạn văn ngắn theo sau bị tách rời):
     - Đoạn văn theo sau ngắn (≤15 từ) → tự động gộp lại vào heading.
     - Đoạn văn dài/mơ hồ → KHÔNG gộp, chỉ ghi warning để review tay.
 7. Per-line filters: dot-leader, pure-dots, orphan-bullet, image placeholder,
    front-matter blacklist, heading level normalize, glued-bullet fix.
 8. Collapse whitespace; trim trailing; collapse blank lines.
 9. Table validation → warnings vào report (không sửa nội dung).

Output mỗi file:
  data/cleaned/<name>.md
  data/cleaned/<name>_report.json

Dùng:
  python -m indexing.cleaner                    # toàn bộ data/interim/
  python -m indexing.cleaner input.md out.md    # 1 file
  python -m indexing.cleaner path/to/dir/       # 1 thư mục
"""

import html
import json
import re
import sys
import unicodedata
import argparse
from pathlib import Path

# Force UTF-8 output trên Windows (tránh UnicodeEncodeError với emoji)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_DIR  = _PROJECT_ROOT / "data" / "interim"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "cleaned"

# ---------------------------------------------------------------------------
# (3) Legacy font fix
# ---------------------------------------------------------------------------
# Empty: 7 file mới qua Docling không có lỗi font VNI/TCVN3.
# Thêm mapping vào đây nếu sau này phát hiện file scan có lỗi font.
LEGACY_FONT_FIX: dict[str, str] = {}

# ---------------------------------------------------------------------------
# (4) PUA character mapping (Symbol / Wingdings)
# ---------------------------------------------------------------------------
# Ký hiệu y khoa quan trọng — ánh xạ đúng, không xóa:
SYMBOL_PUA_MEANINGFUL: dict[str, str] = {
    "\uf0b3": "≥",   # byte 0xB3 trong font Symbol
    "\uf0a3": "≤",   # byte 0xA3 trong font Symbol
    "\uf062": "β",   # byte 0x62 ('b') trong font Symbol → Greek beta
}
# Glyph bullet trang trí — xóa an toàn (đứng ngay sau "- " của Markdown):
SYMBOL_PUA_DECORATIVE: list[str] = ["\uf02b", "\uf0b7"]

_ALL_PUA: set[str] = set(SYMBOL_PUA_MEANINGFUL) | set(SYMBOL_PUA_DECORATIVE)


def _has_pua(text: str) -> bool:
    """Kiểm tra nhanh xem text có chứa PUA cần xử lý không."""
    return any(ch in text for ch in _ALL_PUA)


# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

# Dot-leader THẬT của mục lục: >=8 chấm, kết thúc bằng số/la mã
DOT_LEADER_RE = re.compile(
    r"\.{8,}\s*(\d{1,4}|[ivxlcdm]{1,6})\s*\|?\s*$", re.IGNORECASE
)

# Dòng chỉ toàn dấu chấm (chỗ trống điền bài tập)
PURE_DOTS_RE = re.compile(r"^[\s|]*\.{5,}[\s|]*$")

# STANDALONE_PAGE_NUM_RE — DISABLED.
# Xóa số đứng riêng quá nguy hiểm với bảng y khoa
# (liều thuốc, tuần thai, chỉ số xét nghiệm có thể là số đơn trên 1 ô).
# STANDALONE_PAGE_NUM_RE = re.compile(r"^\s*(\d{1,4}|[ivxlcdm]{1,6})\s*$", re.IGNORECASE)

# Bullet lồng bị dính: "- -Text" / "- - Text" / "- +Text" → "- Text"
# \s* giữa marker và content để bắt cả trường hợp có khoảng trắng sau dấu
GLUED_BULLET_RE = re.compile(r"^(\s*)- ([-+])\s*(\S.*)$")

# Dòng rác "- -" / "- +" đứng riêng (không có nội dung)
ORPHAN_BULLET_RE = re.compile(r"^\s*- [-+]\s*$")

# Heading TOC — chấp nhận # hoặc ##
TOC_HEADING_RE = re.compile(r"^#{1,2}\s*MỤC LỤC\s*$", re.IGNORECASE)

# Heading bất kỳ (dùng để xác định kết thúc block TOC)
HEADING_RE = re.compile(r"^#{1,6}\s+\S")
HEADING_TEXT_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# Suffix heading bị vỡ: heading có 1-3 ký tự thường tiếng Việt/ASCII
BROKEN_HEADING_SUFFIX_RE = re.compile(
    r"^#{1,6}\s+"
    r"([a-záàảãạăắặằẳẵâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]{1,3})"
    r"\s*$",
    re.IGNORECASE | re.UNICODE,
)

# Heading kết thúc bằng dấu câu hoàn chỉnh → KHÔNG coi là bị vỡ
_COMPLETE_END_RE = re.compile(r"[.;:?!]\s*$")

# Heading hierarchy: ## + số thứ tự → chuẩn hóa số lượng #
HEADING_LEVEL_RE = re.compile(r"^##\s+((\d+\.)+\d*\.?)\s+(.*)")

# Bảng Markdown
_TABLE_ROW_RE = re.compile(r"^\s*\|")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-:|]+\|[\s\-:|]*$")

# ---------------------------------------------------------------------------
# Front-matter blacklist (thu hẹp)
# "Ban hành kèm theo Quyết định" ĐÃ BỎ — có thể xuất hiện trong nội dung thật.
# ---------------------------------------------------------------------------
FRONT_MATTER_BLACKLIST: list[re.Pattern] = [
    re.compile(r"^(#{1,2}\s*)?BỘ Y TẾ\s*$", re.IGNORECASE),
    re.compile(r"^\s*---o0o---\s*$", re.IGNORECASE),
    re.compile(r"^(#{1,2}\s*)?CHỦ BIÊN\s*$", re.IGNORECASE),
    re.compile(r"^(#{1,2}\s*)?ĐỒNG CHỦ BIÊN\s*$", re.IGNORECASE),
    re.compile(r"^\s*(PGS\.TS\.|GS\.TS\.|TS\.BS\.|ThS\.BS\.)\s+", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# (6a) Bullet-disguised numbered/roman heading promotion — config
# ---------------------------------------------------------------------------
# Docling/PDF-to-md đôi khi convert heading số La Mã ("I.", "II.", "C." tức
# 1 chữ cái đơn dùng làm nhãn liệt kê A/B/C...) hoặc heading nhiều cấp số
# ("1.2.", "4.4.7.") thành bullet "- " thường, khiến promote_numbered_headings
# (chỉ xét dòng KHÔNG bắt đầu bằng "-") bỏ sót hoàn toàn — không sửa cũng
# không cảnh báo. Xử lý riêng 2 dạng này trước khi vào bước 6b.

# "- I. Text" / "- II. Text" / "- C. Text" (roman numeral 2-4 ký tự hợp lệ,
# HOẶC 1 chữ cái đơn A-Z dùng làm nhãn liệt kê kiểu A./B./C.)
ROMAN_OR_ALPHA_BULLET_RE = re.compile(r"^-\s+([IVXLCDM]{2,4}|[A-Z])\.\s+(\S.*)$")

# "- 1.2. Text" / "- 4.4.7. Text" (>=2 cấp số phân tách bằng dấu chấm — tín
# hiệu heading rất đặc thù của bộ tài liệu này, không trùng bullet nội dung
# thường vì bullet nội dung thật không đánh số nhiều cấp kiểu N.N.)
DOTTED_NUM_BULLET_RE = re.compile(r"^-\s+(\d+(?:\.\d+)+)\.?\s+(\S.*)$")

_SENTENCE_END_RE_EARLY = re.compile(r"[.;:?!]\s*$")
_BULLET_PREFIX_RE_EARLY = re.compile(r"^[-+]\s")


def _prev_is_boundary(lines: list[str], i: int) -> bool:
    """True nếu dòng trước dòng i là dòng trắng, câu hoàn chỉnh, hoặc bullet
    item — tức dòng i không phải phần tiếp nối của 1 câu đang dang dở."""
    prev = lines[i - 1].strip() if i > 0 else ""
    return (
        prev == ""
        or bool(_SENTENCE_END_RE_EARLY.search(prev))
        or bool(_BULLET_PREFIX_RE_EARLY.match(prev))
    )


def promote_bulleted_headings(lines: list[str]) -> tuple[list[str], list[dict]]:
    """Thêm '#' cho heading bị "đội lốt" bullet '- ' (xem giải thích ở config
    phía trên). Cả 2 dạng đều tự động sửa (HIGH confidence) khi đứng ở vị trí
    "đứng riêng" hợp lệ — mẫu số La Mã / nhiều cấp số là tín hiệu đặc thù,
    rủi ro trùng với bullet nội dung thật rất thấp trong bộ tài liệu này.
    """
    result: list[str] = []
    warnings: list[dict] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        lineno = i + 1

        m_dot = DOTTED_NUM_BULLET_RE.match(stripped)
        if m_dot and _prev_is_boundary(lines, i):
            # Giữ nguyên text gốc sau "- ", chỉ đổi marker "- " → "## "
            # (bước 7f phía sau sẽ tự chuẩn hoá lại số lượng '#' theo số cấp).
            new_line = "## " + stripped[2:].lstrip()
            result.append(new_line)
            warnings.append({
                "type": "heading_promoted_bullet_disguised",
                "line": lineno,
                "detail": f"Bullet '-' đội lốt heading nhiều cấp số, đã tự thêm '##': {stripped[:80]}",
            })
            continue

        m_roman = ROMAN_OR_ALPHA_BULLET_RE.match(stripped)
        if m_roman and _prev_is_boundary(lines, i):
            new_line = "## " + stripped[2:].lstrip()
            result.append(new_line)
            warnings.append({
                "type": "heading_promoted_bullet_disguised",
                "line": lineno,
                "detail": f"Bullet '-' đội lốt heading La Mã/chữ cái, đã tự thêm '##': {stripped[:80]}",
            })
            continue

        result.append(line)

    return result, warnings


# ---------------------------------------------------------------------------
# (6b) Numbered heading promotion — config
# ---------------------------------------------------------------------------

# Chỉ xét dòng dạng "N. Text" đứng một mình (KHÔNG phải bullet "-"/"+"/"|",
# KHÔNG phải heading có sẵn) — do quy ước của bộ tài liệu này, list item thật
# trong văn bản luôn có prefix "-" hoặc "+", còn "N. Text" trần trụi chỉ dùng
# cho heading đánh số.
NUMBERED_LINE_RE = re.compile(r"^(\d+)\.\s+(\S.*)$")

# Ngưỡng số từ để coi 1 dòng "N. Text" Title-Case là heading ngắn (MEDIUM),
# thay vì câu văn dài nằm trong danh sách nội dung (LOW → không tự sửa).
MEDIUM_CONFIDENCE_MAX_WORDS = 8

# Bullet/list-item prefix dùng để nhận diện "dòng trước là 1 mục liệt kê hoàn
# chỉnh" — bullet không nhất thiết kết thúc bằng dấu câu, nhưng vẫn là ranh
# giới rõ ràng (không phải câu đang dang dở), nên được coi là "đứng riêng"
# giống dòng trắng / câu kết thúc bằng dấu câu.
# (Fix: trước đây thiếu điều kiện này khiến heading ALL CAPS đứng ngay sau 1
# bullet không kết thúc bằng dấu câu — vd "CHỈ ĐỊNH" sau "- Không tắm ngay
# sau khi trẻ bú" — bị bỏ sót hoàn toàn, không promote cũng không warning.)
_BULLET_PREFIX_RE = re.compile(r"^[-+]\s")


def _is_allcaps_vn(text: str) -> bool:
    """True nếu `text` toàn chữ hoa (Unicode-aware, đúng cho cả Ơ/Ư/Ă/Â/Ê/Ô...).

    Dùng str.isalpha()/upper()/lower() của Python thay vì tự liệt kê range ký
    tự — vì chữ Việt mở rộng (Ơ, Ư...) nằm ngoài khối Latin-1 (À-Ỹ) nên regex
    charset thủ công dễ bỏ sót.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    letters_str = "".join(letters)
    return letters_str == letters_str.upper() and letters_str != letters_str.lower()


_SENTENCE_END_RE = re.compile(r"[.;:?!]\s*$")


def promote_numbered_headings(lines: list[str]) -> tuple[list[str], list[dict]]:
    """Thêm '#' cho heading đánh số bị Docling bỏ sót.

    Chỉ xét dòng "N. Text" đứng riêng (dòng trước là dòng trắng, câu hoàn
    chỉnh, hoặc 1 bullet item — tức không phải câu đang dang dở), không phải
    bullet/table/heading có sẵn.

    - ALL CAPS               → HIGH confidence  → tự thêm '## '.
    - Title Case, ≤8 từ      → MEDIUM confidence → tự thêm '### '.
    - Còn lại (câu dài, ...) → LOW confidence    → KHÔNG sửa, chỉ cảnh báo.

    Điều kiện "đứng riêng" của dòng trước:
      - Tier MEDIUM: bắt buộc dòng ngay trước là dòng trắng (chặt, để không
        nuốt nhầm list item thường giữa 1 đoạn liệt kê).
      - Tier ALL CAPS và tier LOW (đã có tín hiệu đủ mạnh, hoặc chỉ để cảnh
        báo chứ không tự sửa): chấp nhận dòng trước là dòng trắng, HOẶC kết
        thúc bằng dấu câu hoàn chỉnh (., ;, :, ?, !), HOẶC là 1 bullet item
        ("- ...", "+ ...") — vì Docling đôi khi làm mất dòng trắng trước
        heading loại này, và bullet không nhất thiết kết thúc bằng dấu câu
        nhưng vẫn là 1 đơn vị hoàn chỉnh (không phải câu dang dở).

    Mọi thay đổi đều được ghi vào warnings để audit lại (grep theo 'type').
    """
    result: list[str] = []
    warnings: list[dict] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        m = NUMBERED_LINE_RE.match(stripped)

        if not m or stripped.startswith(("-", "+", "|", "#")):
            result.append(line)
            continue

        prev = lines[i - 1].strip() if i > 0 else ""
        prev_blank = prev == ""
        prev_sentence_end = bool(_SENTENCE_END_RE.search(prev))
        prev_is_bullet = bool(_BULLET_PREFIX_RE.match(prev))
        prev_is_boundary = prev_blank or prev_sentence_end or prev_is_bullet

        rest = m.group(2)
        word_count = len(rest.split())
        lineno = i + 1

        if _is_allcaps_vn(rest) and prev_is_boundary:
            result.append(f"## {stripped}")
            warnings.append({
                "type": "heading_promoted_high_confidence",
                "line": lineno,
                "detail": f"ALL CAPS, đã tự thêm '##': {stripped[:80]}",
            })
        elif prev_blank and word_count <= MEDIUM_CONFIDENCE_MAX_WORDS and rest.count(".") <= 1:
            result.append(f"### {stripped}")
            warnings.append({
                "type": "heading_promoted_medium_confidence",
                "line": lineno,
                "detail": f"Title Case ngắn, đã tự thêm '###' (nên spot-check): {stripped[:80]}",
            })
        elif prev_is_boundary:
            result.append(line)
            warnings.append({
                "type": "heading_candidate_low_confidence_needs_review",
                "line": lineno,
                "detail": f"Có thể là heading thiếu '#' nhưng câu dài/mơ hồ, GIỮ NGUYÊN, cần review tay: {stripped[:80]}",
            })
        else:
            result.append(line)

    return result, warnings


# ---------------------------------------------------------------------------
# (6c) Dangling-connector heading repair — config
# ---------------------------------------------------------------------------

DANGLING_CONNECTOR_WORDS: set[str] = {
    "và", "của", "trong", "là", "để", "với", "theo", "khi", "cho", "hoặc",
    "có", "từ", "do", "các", "những", "đến", "sau", "trên", "dưới", "về",
    "như", "nếu",
}

# Đoạn văn nối tiếp phải đủ NGẮN mới tự động gộp — dài hơn thì có thể là cả
# 1 đoạn nội dung riêng biệt (không nên nuốt vào heading).
DANGLING_MERGE_MAX_WORDS = 15


def _ends_dangling(text: str) -> bool:
    text = text.strip()
    if text.endswith("-"):
        return True
    tokens = re.split(r"\s+", text)
    if not tokens:
        return False
    last = tokens[-1].lower().strip(".,:;")
    return last in DANGLING_CONNECTOR_WORDS


def repair_dangling_connector_headings(lines: list[str]) -> tuple[list[str], list[dict]]:
    """Gộp heading bị vỡ khi phần còn lại của câu bị tách thành đoạn văn riêng.

    Khác với repair_broken_headings (vỡ NGANG do PDF wrap giữa từ, 2 dòng liền
    kề, suffix 1-3 ký tự): ở đây heading kết thúc bằng '-' hoặc 1 từ nối lửng
    lơ (để, và, theo...), và phần còn lại nằm sau MỘT HAY NHIỀU dòng trắng,
    dưới dạng đoạn văn riêng (không phải bullet/heading/table).

    Chỉ tự động gộp khi đoạn văn theo sau đủ NGẮN (≤ DANGLING_MERGE_MAX_WORDS
    từ) — nếu dài/mơ hồ thì KHÔNG động vào, chỉ ghi warning.
    """
    result: list[str] = []
    warnings: list[dict] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        m = HEADING_TEXT_RE.match(line)

        if not m or not _ends_dangling(m.group(2)):
            result.append(line)
            i += 1
            continue

        # tìm dòng nội dung tiếp theo, bỏ qua các dòng trắng
        j = i + 1
        while j < n and lines[j].strip() == "":
            j += 1

        if j >= n or j == i + 1:
            # không có dòng trắng phân cách, hoặc hết file → không phải case này
            result.append(line)
            i += 1
            continue

        nxt = lines[j].strip()
        is_plain_paragraph = nxt and not nxt.startswith(("#", "-", "+", "|"))
        short_enough = len(nxt.split()) <= DANGLING_MERGE_MAX_WORDS

        if is_plain_paragraph and short_enough:
            merged = line.rstrip() + " " + nxt
            result.append(merged)
            warnings.append({
                "type": "heading_dangling_connector_merged",
                "line": i + 1,
                "detail": f"Đã gộp heading với đoạn văn nối tiếp: '{m.group(2)[:50]}' + '{nxt[:50]}'",
            })
            i = j + 1
            continue

        # mơ hồ / đoạn văn quá dài → không tự gộp, chỉ cảnh báo
        result.append(line)
        warnings.append({
            "type": "heading_dangling_connector_not_merged",
            "line": i + 1,
            "detail": f"Heading có vẻ bị vỡ ('{m.group(2)[:50]}') nhưng đoạn theo sau dài/không rõ, GIỮ NGUYÊN, cần review tay",
        })
        i += 1

    return result, warnings


# ---------------------------------------------------------------------------
# (5) TOC removal — conservative
# ---------------------------------------------------------------------------

def remove_toc_block(lines: list[str]) -> list[str]:
    """Xóa khối '## MỤC LỤC' → heading tiếp theo.

    Chỉ xóa khi block giữa 2 heading đó chứa ÍT NHẤT 1 dòng dot-leader,
    xác nhận đây thực sự là mục lục (không phải section tên 'Mục lục').
    """
    result: list[str] = []
    i = 0
    while i < len(lines):
        if TOC_HEADING_RE.match(lines[i].strip()):
            j = i + 1
            while j < len(lines) and not HEADING_RE.match(lines[j]):
                j += 1
            block = lines[i + 1 : j]
            if any(DOT_LEADER_RE.search(ln) for ln in block):
                i = j
                continue
        result.append(lines[i])
        i += 1
    return result


# ---------------------------------------------------------------------------
# (6) Broken heading repair — strict (adjacent lines only)
# ---------------------------------------------------------------------------

def repair_broken_headings(lines: list[str]) -> list[str]:
    """Gộp heading bị vỡ ngang do PDF xuống dòng.

    Điều kiện gộp (TẤT CẢ phải đúng):
      1. Dòng hiện tại là heading.
      2. Dòng NGAY KẾ (không qua dòng trắng) khớp BROKEN_HEADING_SUFFIX_RE.
      3. Heading hiện tại KHÔNG kết thúc dấu câu hoàn chỉnh (. ; : ? !).
    """
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        next_idx = i + 1

        if (
            re.match(r"^#{1,6}\s+", line)
            and next_idx < len(lines)
            and BROKEN_HEADING_SUFFIX_RE.match(lines[next_idx])
            and not _COMPLETE_END_RE.search(line.rstrip())
        ):
            suffix = BROKEN_HEADING_SUFFIX_RE.match(lines[next_idx]).group(1)
            result.append(line.rstrip() + suffix)
            i = next_idx + 1
            continue

        result.append(line)
        i += 1
    return result


# ---------------------------------------------------------------------------
# (9) Table validation
# ---------------------------------------------------------------------------

def _count_cols(row: str) -> int:
    """Đếm số ô trong 1 hàng bảng Markdown."""
    return len(row.strip().strip("|").split("|"))


def validate_tables(lines: list[str]) -> list[dict]:
    """Tìm bảng Markdown và cảnh báo khi cấu trúc không nhất quán.

    Kiểm tra:
    - Separator có đúng số cột với header không.
    - Mỗi data row có đúng số cột với header không.

    Không bao giờ sửa nội dung — chỉ trả về danh sách warning.
    """
    warnings: list[dict] = []
    i = 0
    while i < len(lines):
        if not _TABLE_ROW_RE.match(lines[i]):
            i += 1
            continue

        header_cols   = _count_cols(lines[i])
        header_lineno = i + 1  # 1-indexed

        sep_idx = i + 1
        if sep_idx >= len(lines) or not _TABLE_SEP_RE.match(lines[sep_idx]):
            i += 1
            continue

        sep_cols = _count_cols(lines[sep_idx])
        if sep_cols != header_cols:
            warnings.append({
                "type": "table_separator_col_mismatch",
                "line": sep_idx + 1,
                "detail": (
                    f"Header: {header_cols} cols, "
                    f"Separator: {sep_cols} cols "
                    f"(header tại dòng {header_lineno})"
                ),
            })

        j = sep_idx + 1
        while j < len(lines) and _TABLE_ROW_RE.match(lines[j]):
            row_cols = _count_cols(lines[j])
            if row_cols != header_cols:
                warnings.append({
                    "type": "table_row_col_mismatch",
                    "line": j + 1,
                    "detail": (
                        f"Header: {header_cols} cols, "
                        f"Row: {row_cols} cols "
                        f"(header tại dòng {header_lineno})"
                    ),
                })
            j += 1

        i = j
    return warnings


# ---------------------------------------------------------------------------
# Main clean function
# ---------------------------------------------------------------------------

def clean_markdown(
    text: str,
    drop_image_placeholders: bool = True,
) -> tuple[str, list[dict]]:
    """Làm sạch chuỗi Markdown.

    Returns:
        (cleaned_text, warnings)  — warnings dùng để ghi validation_report.json.
    """
    # (0) Chuẩn hóa line ending: CRLF/CR → LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # (1) Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # (2) HTML entities
    text = html.unescape(text)

    # (3) Legacy font fix (hiện trống)
    for wrong, right in LEGACY_FONT_FIX.items():
        text = text.replace(wrong, right)

    # (4) PUA — chỉ khi file thực sự chứa PUA
    if _has_pua(text):
        for wrong, right in SYMBOL_PUA_MEANINGFUL.items():
            text = text.replace(wrong, right)
        for ch in SYMBOL_PUA_DECORATIVE:
            text = text.replace(ch, "")

    lines = text.split("\n")

    # (5) Xóa khối MỤC LỤC (bảo thủ)
    lines = remove_toc_block(lines)

    # (6) Sửa heading bị vỡ (chỉ dòng liền kề)
    lines = repair_broken_headings(lines)

    # (6a) Thêm '#' cho heading bị "đội lốt" bullet '- ' (roman/alpha/nhiều cấp số)
    lines, bulleted_heading_warnings = promote_bulleted_headings(lines)

    # (6b) Thêm '#' cho heading đánh số bị bỏ sót
    lines, heading_promote_warnings = promote_numbered_headings(lines)

    # (6c) Gộp heading bị vỡ nối với đoạn văn phía sau (dấu '-'/liên từ lửng lơ)
    lines, dangling_warnings = repair_dangling_connector_headings(lines)

    # (9) Validate bảng TRƯỚC khi lọc dòng
    table_warnings = validate_tables(lines)

    all_warnings = (
        bulleted_heading_warnings
        + heading_promote_warnings
        + dangling_warnings
        + table_warnings
    )

    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.strip()

        # (7a) Image placeholder
        if drop_image_placeholders and stripped == "<!-- image -->":
            continue

        # (7b) Dot-leader mục lục
        if DOT_LEADER_RE.search(line):
            continue

        # (7c) Dòng chỉ toàn dấu chấm
        if PURE_DOTS_RE.match(line):
            continue

        # STANDALONE_PAGE_NUM_RE — DISABLED

        # (7d) Orphan bullet
        if ORPHAN_BULLET_RE.match(line):
            continue

        # (7e) Front-matter blacklist
        if any(regex.match(line) for regex in FRONT_MATTER_BLACKLIST):
            continue

        # (7f) Chuẩn hóa cấp heading: ## 3.1. → ###
        m = HEADING_LEVEL_RE.match(line)
        if m:
            prefix = m.group(1)
            rest   = m.group(3)
            nums   = [x for x in prefix.split(".") if x.isdigit()]
            level  = min(len(nums) + 1, 6)
            line   = f"{'#' * level} {prefix} {rest}"

        # (7g) Bullet lồng bị dính
        m = GLUED_BULLET_RE.match(line)
        if m:
            indent, _marker, content = m.groups()
            line = f"{indent}- {content}"

        # (8a) Gộp khoảng trắng kép
        line = re.sub(r"[ \t]{2,}", " ", line)

        # (8b) Trailing whitespace
        line = line.rstrip()

        cleaned_lines.append(line)

    # (8c) Gộp nhiều dòng trống liên tiếp → tối đa 1
    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip() + "\n", all_warnings


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def process_file(input_path: Path, output_path: Path) -> None:
    """Làm sạch 1 file Markdown và ghi ra output_path + _report.json."""
    text = input_path.read_text(encoding="utf-8")
    before_len = len(text)

    cleaned, warnings = clean_markdown(text)
    after_len = len(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # FIX (CRLF regression): output_path.write_text() ở text mode mặc định
    # tự dịch '\n' → os.linesep khi ghi (trên Windows là '\r\n'), bất kể nội
    # dung `cleaned` đã chuẩn hoá thành LF thuần ở bước (0). Dùng write_bytes
    # để ghi byte thô, không qua bất kỳ dịch newline nào — đảm bảo output
    # LUÔN là LF thuần, chạy giống nhau trên Windows/Linux/macOS.
    output_path.write_bytes(cleaned.encode("utf-8"))

    reduction = (1 - after_len / before_len) * 100 if before_len else 0.0

    report = {
        "source": input_path.name,
        "chars_before": before_len,
        "chars_after": after_len,
        "reduction_pct": round(reduction, 2),
        "warnings": warnings,
    }
    report_path = output_path.with_name(output_path.stem + "_report.json")
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    # Cùng lý do như trên: ghi bytes thô để report.json cũng luôn LF thuần.
    report_path.write_bytes((report_json + "\n").encode("utf-8"))

    warn_label = f"⚠️  {len(warnings)} warning(s)" if warnings else "✅ no warnings"
    print(f"✅ {input_path.name}")
    print(f"   {before_len:,} ký tự → {after_len:,} ký tự (giảm {reduction:.1f}%) | {warn_label}")
    print(f"   Clean  : {output_path}")
    print(f"   Report : {report_path}")


def process_dir(input_dir: Path, output_dir: Path) -> None:
    """Làm sạch toàn bộ *.md trong input_dir, lưu vào output_dir."""
    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        print(f"[cleaner] ⚠️  Không tìm thấy file .md nào trong: {input_dir}")
        return

    print(f"[cleaner] 📂 Xử lý {len(md_files)} file từ : {input_dir}")
    print(f"[cleaner] 📁 Output              → {output_dir}\n")

    success: list[str] = []
    failed:  list[dict] = []

    for md_file in md_files:
        try:
            process_file(md_file, output_dir / md_file.name)
            success.append(md_file.name)
        except Exception as e:
            failed.append({"file": md_file.name, "error": str(e)})
            print(f"[cleaner] ❌ Lỗi: {md_file.name}")
            print(f"          {e}")

    print(f"\n{'=' * 60}")
    print(f"🎉 Thành công : {len(success)}")
    print(f"❌ Thất bại   : {len(failed)}")
    if failed:
        print("\nCác file lỗi:")
        for item in failed:
            print(f"  - {item['file']}: {item['error']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Làm sạch Markdown xuất từ Docling, chuẩn bị cho chunking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input", nargs="?",
        help=(
            "File .md hoặc thư mục chứa các file .md cần xử lý. "
            f"Mặc định: {DEFAULT_INPUT_DIR}"
        ),
    )
    parser.add_argument(
        "output", nargs="?",
        help=(
            "File .md đầu ra (khi input là file) hoặc thư mục đầu ra "
            f"(khi input là thư mục). Mặc định: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    args = parser.parse_args()

    input_path  = Path(args.input)  if args.input  else DEFAULT_INPUT_DIR
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR

    if input_path.is_dir():
        process_dir(input_path, output_path)
    elif input_path.is_file():
        if output_path.suffix != ".md":
            output_path = output_path / input_path.name
        process_file(input_path, output_path)
    else:
        print(f"[cleaner] ❌ Không tìm thấy: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()