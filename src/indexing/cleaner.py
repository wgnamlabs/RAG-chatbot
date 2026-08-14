"""
Làm sạch file Markdown xuất ra từ Docling, chuẩn bị cho bước chunking.

Các bước xử lý (thứ tự có ý nghĩa):
1. Chuẩn hóa Unicode về NFC.
2. Sửa lỗi font cũ: 'Ƣ/ƣ' (LATIN LETTER OI) dùng nhầm thay cho 'Ư/ư'.
3. Sửa ký tự vùng Private Use Area (PUA) do font Symbol/Wingdings gây ra
   khi PDF không có ToUnicode CMap chuẩn. Một số PUA là ký hiệu y khoa quan
   trọng (≥, ≤, β) PHẢI được ánh xạ đúng chứ không được xóa; một số khác chỉ
   là glyph bullet trang trí (●, +) đứng ngay sau "- " nên xóa bỏ an toàn.
4. Xóa nguyên khối "## MỤC LỤC" (từ heading đến heading tiếp theo) - đây là
   cách an toàn nhất để loại bỏ mục lục dạng bảng dot-leader mà không đụng
   tới nội dung thật.
5. Xóa các dòng dot-leader CÒN SÓT ngoài khối trên - CHỈ khi dòng đó kết
   thúc bằng số trang/số la mã ngay sau các dấu chấm (đúng định dạng mục
   lục), để KHÔNG xóa nhầm các câu văn thật kết thúc bằng dấu chấm lửng
   kiểu "...vi khuẩn, ký sinh trùng.....".
6. Xóa các dòng chỉ toàn dấu chấm (chỗ trống điền bài tập, không mang
   thông tin).
7. Sửa lỗi bullet lồng bị Docling dính liền: "- -Text" / "- +Text" ->
   "- Text". Regex được ràng buộc chặt để KHÔNG khớp nhầm các dòng phân
   cách trang trí kiểu "---o0o---".
8. Xóa các dòng rác chỉ có "- -" hoặc "- +" đứng riêng (mảnh vỡ list lồng).
9. Xóa các dòng chỉ chứa số trang / số la mã đơn lẻ (rác header-footer PDF).
10. Gộp khoảng trắng kép/nhiều thành 1 khoảng trắng.
11. Gộp nhiều dòng trống liên tiếp thành tối đa 1 dòng trống.
12. (Tùy chọn) Bỏ placeholder "<!-- image -->" vì không mang nội dung text.
13. Xóa khoảng trắng thừa cuối mỗi dòng.

Đường dẫn mặc định (chạy như script):
  Input : data/interim/   (Markdown thô do loader.py xuất ra)
  Output: data/cleaned/   (Markdown đã làm sạch, sẵn sàng cho chunking)

Dùng:
  python -m indexing.cleaner                        # xử lý toàn bộ data/interim/
  python -m indexing.cleaner input.md output.md     # 1 file cụ thể
  python -m indexing.cleaner path/to/dir/           # 1 thư mục tùy chọn
"""

import html
import re
import sys
import unicodedata
import argparse
from pathlib import Path

# --- Đường dẫn mặc định (tính từ gốc dự án) ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_DIR  = _PROJECT_ROOT / "data" / "interim"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "cleaned"

# --- (2) Bảng sửa lỗi font cũ (VNI/TCVN3 không có ToUnicode chuẩn) ---
LEGACY_FONT_FIX = {
    "Ƣ": "Ư",
    "ƣ": "ư",
}

# --- (3) Bảng sửa ký tự PUA của font Symbol/Wingdings ---
# Ký hiệu y khoa QUAN TRỌNG - phải ánh xạ đúng, không được xóa:
SYMBOL_PUA_MEANINGFUL = {
    "\uf0b3": "≥",   # byte 0xB3 trong font Symbol
    "\uf0a3": "≤",   # byte 0xA3 trong font Symbol
    "\uf062": "β",   # byte 0x62 ('b') trong font Symbol -> chữ Hy Lạp beta
}
# Glyph bullet trang trí - luôn đứng ngay sau "- " (markdown đã có bullet
# rồi) nên xóa bỏ an toàn, không mất thông tin:
SYMBOL_PUA_DECORATIVE = ["\uf02b", "\uf0b7"]

# Regex khớp dot-leader THẬT của mục lục: dấu chấm dài (>=8) và PHẢI kết
# thúc dòng bằng số trang/số la mã (có thể còn trong ô bảng | ... |).
DOT_LEADER_RE = re.compile(r"\.{8,}\s*(\d{1,4}|[ivxlcdm]{1,6})\s*\|?\s*$", re.IGNORECASE)

# Dòng chỉ toàn dấu chấm (chỗ trống điền bài tập) - không mang thông tin
PURE_DOTS_RE = re.compile(r"^[\s|]*\.{5,}[\s|]*$")

# Dòng chỉ chứa số trang / số la mã đơn lẻ (rác header-footer)
STANDALONE_PAGE_NUM_RE = re.compile(r"^\s*(\d{1,4}|[ivxlcdm]{1,6})\s*$", re.IGNORECASE)

# Bullet lồng bị dính: yêu cầu CHÍNH XÁC "- " rồi ngay lập tức "-" hoặc "+"
# rồi nội dung. Không khớp nhầm "---o0o---".
GLUED_BULLET_RE = re.compile(r"^(\s*)- ([-+])(\S.*)$")

# Dòng rác chỉ có "- -" hoặc "- +" (không có nội dung theo sau)
ORPHAN_BULLET_RE = re.compile(r"^\s*- [-+]\s*$")

# Heading "## MỤC LỤC" để xóa nguyên khối tới heading tiếp theo
TOC_HEADING_RE = re.compile(r"^##\s*MỤC LỤC\s*$", re.IGNORECASE)
HEADING_RE = re.compile(r"^##\s+\S")

# Match '## ' followed by numbers like '1.', '1.1', '1.1.', '1.1.1', etc.
HEADING_LEVEL_RE = re.compile(r"^##\s+((\d+\.)+\d*\.?)\s+(.*)")

# Heading bị vỡ do OCR ngắt trang: phần tiếp theo chỉ có 1-3 ký tự thường
# Ví dụ: "### 6.3. Điều trị bằng thuố" + "## c" → gộp thành 1 heading
BROKEN_HEADING_SUFFIX_RE = re.compile(r"^#{1,6}\s+([a-záàảãạăắặằẳẵâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]{1,3})\s*$", re.IGNORECASE | re.UNICODE)

# Blacklist lọc rác front-matter
FRONT_MATTER_BLACKLIST = [
    re.compile(r"^(##\s*)?BỘ Y TẾ\s*$", re.IGNORECASE),
    re.compile(r"^\s*---o0o---\s*$", re.IGNORECASE),
    re.compile(r"^(##\s*)?CHỦ BIÊN\s*$", re.IGNORECASE),
    re.compile(r"^(##\s*)?Ban hành kèm theo Quyết định", re.IGNORECASE),
    re.compile(r"^(##\s*)?ĐỒNG CHỦ BIÊN\s*$", re.IGNORECASE),
    re.compile(r"^\s*(PGS\.TS\.|GS\.TS\.|TS\.BS\.|ThS\.BS\.)\s+", re.IGNORECASE),
]


def remove_toc_block(lines: list[str]) -> list[str]:
    """Xóa nguyên khối từ '## MỤC LỤC' tới heading '##' tiếp theo (không bao gồm)."""
    result = []
    i = 0
    while i < len(lines):
        if TOC_HEADING_RE.match(lines[i].strip()):
            i += 1
            while i < len(lines) and not HEADING_RE.match(lines[i]):
                i += 1
            continue
        result.append(lines[i])
        i += 1
    return result


def repair_broken_headings(lines: list[str]) -> list[str]:
    """Gộp heading bị vỡ ngang do OCR ngắt trang.

    Trường hợp điển hình:
        Dòng N  : "### 6.3. Điều trị bằng thuố"   ← heading cụt, không kết thúc hợp lý
        Dòng N+1: "## c"                            ← heading giả, chỉ là đoạn cuối
    Sau sửa    : "### 6.3. Điều trị bằng thuốc"    ← heading hoàn chỉnh

    Điều kiện để gộp:
      - Dòng hiện tại là heading (bắt đầu bằng #+ space).
      - Dòng tiếp theo khớp BROKEN_HEADING_SUFFIX_RE (heading giả 1-3 ký tự thường).
      - Dòng giữa (nếu có dòng trắng) sẽ không gộp — cần liền nhau hoặc cách 1 dòng trắng.
    """
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Kiểm tra xem dòng kế (bỏ qua dòng trắng đơn) có phải heading cụt không
        next_real = i + 1
        # Cho phép tối đa 1 dòng trắng giữa 2 phần vỡ
        if next_real < len(lines) and lines[next_real].strip() == "":
            next_real += 1
        if (
            re.match(r"^#{1,6}\s+", line)                          # dòng hiện tại là heading
            and next_real < len(lines)
            and BROKEN_HEADING_SUFFIX_RE.match(lines[next_real])   # dòng kế là heading giả
        ):
            # Lấy phần text thật của heading giả (không có #)
            suffix_text = BROKEN_HEADING_SUFFIX_RE.match(lines[next_real]).group(1)
            merged = line.rstrip() + suffix_text
            result.append(merged)
            i = next_real + 1  # bỏ qua dòng trắng lẫn heading giả
            continue
        result.append(line)
        i += 1
    return result


def clean_markdown(text: str, drop_image_placeholders: bool = True) -> str:
    """Làm sạch chuỗi Markdown, trả về chuỗi đã xử lý."""
    # (1) Chuẩn hóa Unicode
    text = unicodedata.normalize("NFC", text)

    # (2) Giải mã HTML entities (&lt; → <, &gt; → >, &amp; → &, &#160; → ' ', v.v.)
    text = html.unescape(text)

    # (3) Sửa lỗi font cũ
    for wrong, right in LEGACY_FONT_FIX.items():
        text = text.replace(wrong, right)

    # (4) Sửa PUA
    for wrong, right in SYMBOL_PUA_MEANINGFUL.items():
        text = text.replace(wrong, right)
    for ch in SYMBOL_PUA_DECORATIVE:
        text = text.replace(ch, "")

    lines = text.split("\n")

    # (5) Xóa khối MỤC LỤC
    lines = remove_toc_block(lines)

    # (6) Sửa heading bị vỡ do OCR ngắt trang
    lines = repair_broken_headings(lines)

    cleaned_lines = []
    for line in lines:
        stripped = line.strip()

        # (12) Bỏ image placeholder
        if drop_image_placeholders and stripped == "<!-- image -->":
            continue

        # (5) Dot-leader mục lục
        if DOT_LEADER_RE.search(line):
            continue

        # (6) Dòng chỉ toàn dấu chấm
        if PURE_DOTS_RE.match(line):
            continue

        # (9) Số trang / số la mã đơn lẻ
        if STANDALONE_PAGE_NUM_RE.match(line):
            continue

        # (8) Dòng rác "- -" / "- +"
        if ORPHAN_BULLET_RE.match(line):
            continue

        # (14) Xóa rác front-matter
        if any(regex.match(line) for regex in FRONT_MATTER_BLACKLIST):
            continue

        # (15) Khôi phục cấp heading nhiều tầng (vd: 3.1. -> ###)
        m = HEADING_LEVEL_RE.match(line)
        if m:
            prefix = m.group(1) # e.g. "3.1."
            rest = m.group(3)
            nums = [x for x in prefix.split('.') if x.isdigit()]
            level = len(nums)
            hashes = '#' * min(level + 1, 6)
            line = f"{hashes} {prefix} {rest}"

        # (7) Bullet lồng bị dính
        m = GLUED_BULLET_RE.match(line)
        if m:
            indent, _marker, content = m.groups()
            line = f"{indent}- {content}"

        # (10) Gộp khoảng trắng kép
        line = re.sub(r"[ \t]{2,}", " ", line)

        # (13) Xóa trailing whitespace
        line = line.rstrip()

        cleaned_lines.append(line)

    # (11) Gộp nhiều dòng trống liên tiếp
    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip() + "\n"


def process_file(input_path: Path, output_path: Path) -> None:
    """Làm sạch 1 file Markdown và ghi ra output_path."""
    text = input_path.read_text(encoding="utf-8")
    before_len = len(text)

    cleaned = clean_markdown(text)
    after_len = len(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")

    reduction = (1 - after_len / before_len) * 100 if before_len else 0
    print(f"✅ {input_path.name}")
    print(f"   {before_len:,} ký tự -> {after_len:,} ký tự (giảm {reduction:.1f}%)")
    print(f"   Lưu tại: {output_path}")


def process_dir(input_dir: Path, output_dir: Path) -> None:
    """Làm sạch toàn bộ *.md trong input_dir, lưu vào output_dir."""
    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        print(f"[cleaner] ⚠️  Không tìm thấy file .md nào trong: {input_dir}")
        return

    print(f"[cleaner] 📂 Xử lý {len(md_files)} file từ: {input_dir}")
    print(f"[cleaner] 📁 Output   -> {output_dir}\n")

    for md_file in md_files:
        process_file(md_file, output_dir / md_file.name)

    print(f"\n[cleaner] ✅ Hoàn thành — {len(md_files)} file đã làm sạch.")


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
        # Nếu output là thư mục (hoặc chưa tồn tại nhưng không có đuôi .md),
        # tự ghép tên file vào.
        if output_path.suffix != ".md":
            output_path = output_path / input_path.name
        process_file(input_path, output_path)
    else:
        print(f"[cleaner] ❌ Không tìm thấy: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
