"""
Module loader xử lý việc trích xuất văn bản từ file PDF bằng thư viện Docling.

THAY ĐỔI so với bản gốc:
- Hỗ trợ cấu hình force_ocr RIÊNG cho từng file PDF (qua pdf_ocr_config.py),
  thay vì 1 flag --force_ocr chung áp dụng cho cả batch.
- Với 3 file bạn cung cấp, cả 3 đều có text layer sẵn (không phải bản scan)
  nên mặc định force_ocr=False cho tất cả (xem pdf_ocr_config.py để chỉnh
  từng file nếu sau này có file scan cần OCR).
"""

import sys
import argparse
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.document_converter import DocumentConverter, PdfFormatOption
import torch

from pdf_ocr_config import get_force_ocr, FORCE_OCR_DEFAULT

# --- In thông tin phần cứng ngay khi load module ---
_use_gpu = torch.cuda.is_available()
_device_name = torch.cuda.get_device_name(0) if _use_gpu else "CPU"
print(f"[loader] 💻 Thiết bị sử dụng: {'CUDA ✅ ' + _device_name if _use_gpu else 'CPU ⚠️'}")

# Cache converter theo từng giá trị force_ocr (vì OCR options khác nhau
# cần pipeline khác nhau -> không thể dùng chung 1 converter cho cả True/False)
_converters: dict[bool, DocumentConverter] = {}


def get_converter(force_ocr: bool = False, languages: list[str] = None) -> DocumentConverter:
    """Tạo và cache DocumentConverter theo từng cấu hình force_ocr (model được load 1 lần/cấu hình, tái sử dụng cho cả batch)."""
    if force_ocr in _converters:
        return _converters[force_ocr]

    languages = languages or ["vi", "en"]

    print(f"[loader] 🔧 Đang khởi tạo pipeline Docling (force_ocr={force_ocr})...")
    t0 = time.time()

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = force_ocr
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    pipeline_options.ocr_options = EasyOcrOptions(
        lang=languages,
        force_full_page_ocr=force_ocr,   # tương đương --force_ocr bên Marker
        use_gpu=_use_gpu,
    )

    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CUDA if _use_gpu else AcceleratorDevice.CPU,
        num_threads=8,
    )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    _converters[force_ocr] = converter
    print(f"[loader] ✅ Pipeline sẵn sàng sau {time.time()-t0:.1f}s")
    return converter


def load_pdf(pdf_path: str, force_ocr: bool = False, languages: list[str] = None) -> dict:
    """Load 1 file PDF bằng Docling, trả về markdown + metadata."""
    print(f"[loader] Đang convert {pdf_path} — force_ocr={force_ocr}")

    converter = get_converter(force_ocr=force_ocr, languages=languages)
    result = converter.convert(pdf_path)
    markdown = result.document.export_to_markdown()

    return {
        "markdown": markdown,
        "metadata": {
            "num_pages": result.document.num_pages(),
            "status": str(result.status),
        },
    }


def _process_one(pdf_path: Path, output_dir: Path, force_ocr: bool, languages: list[str]) -> None:
    """Xử lý 1 file PDF và lưu kết quả Markdown vào output_dir."""
    result = load_pdf(str(pdf_path), force_ocr=force_ocr, languages=languages)
    md_text = result["markdown"]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print("=" * 60)
    print(f"✅ Đã trích xuất xong và lưu kết quả tại: {output_path}")
    print(f"   Số trang: {result['metadata']['num_pages']}")
    print(f"   Tổng số ký tự: {len(md_text)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF loader dùng Docling")
    parser.add_argument("pdf_path", help="Đường dẫn file PDF hoặc thư mục chứa các file PDF")
    parser.add_argument("--force_ocr", action="store_true",
                         help="Ép OCR toàn bộ trang cho TẤT CẢ file, bất kể cấu hình trong pdf_ocr_config.py "
                              "(dùng khi muốn override toàn bộ, ví dụ debug)")
    parser.add_argument("--languages", nargs="+", default=["vi", "en"],
                         help="Danh sách ngôn ngữ OCR, mặc định: vi en")
    parser.add_argument("--only-new", action="store_true", dest="only_new",
                         help="Bỏ qua file đã có output trong data/interim/")
    args = parser.parse_args()

    input_path = Path(args.pdf_path)
    output_dir = Path("data/interim")

    def resolve_force_ocr(pdf_file: Path) -> bool:
        """Ưu tiên: nếu người dùng truyền --force_ocr trên CLI thì ép True cho hết.
        Ngược lại, tra cứu theo từng filename trong pdf_ocr_config.py."""
        if args.force_ocr:
            return True
        return get_force_ocr(pdf_file.name)

    if input_path.is_dir():
        pdf_files = sorted(input_path.glob("*.pdf"))
        if not pdf_files:
            print(f"[loader] Không tìm thấy file PDF nào trong: {input_path}")
            sys.exit(1)

        to_skip = {
            f for f in pdf_files
            if args.only_new and (output_dir / f"{f.stem}.md").exists()
        }
        to_process = [f for f in pdf_files if f not in to_skip]
        total = len(to_process)
        processed = []

        for pdf_file in pdf_files:
            if pdf_file in to_skip:
                print(f"[loader] ⏭  Bỏ qua (đã có output): {pdf_file.name}")
                continue
            file_force_ocr = resolve_force_ocr(pdf_file)
            print(f"\n[loader] 📄 Đang xử lý ({len(processed)+1}/{total}): {pdf_file.name} "
                  f"(force_ocr={file_force_ocr})")
            _process_one(pdf_file, output_dir, file_force_ocr, args.languages)
            processed.append(pdf_file.name)

        print("\n" + "=" * 60)
        print(f"🎉 Hoàn tất batch! Đã xử lý: {len(processed)} file — Bỏ qua: {len(to_skip)} file")
        print("=" * 60)

    elif input_path.is_file():
        file_force_ocr = resolve_force_ocr(input_path)
        _process_one(input_path, output_dir, file_force_ocr, args.languages)

    else:
        print(f"[loader] ❌ Không tìm thấy file hoặc thư mục: {input_path}")
        sys.exit(1)
