"""
Module loader xử lý việc trích xuất văn bản từ file PDF bằng thư viện Docling.

THAY ĐỔI so với bản gốc:
- Hỗ trợ cấu hình force_ocr RIÊNG cho từng file PDF (qua OCR_CONFIG bên dưới),
  thay vì 1 flag --force_ocr chung áp dụng cho cả batch.
- Với 7 file hiện tại đều có text layer sẵn nên FORCE_OCR_DEFAULT = False.
- Cache converter theo (force_ocr, languages) — tránh bug khi gọi với languages khác nhau.
- Output: raw Markdown + metadata JSON (không clean trong bước này).

Pipeline:
    loader.py  →  raw .md + .json  →  cleaner.py  →  chunker.py  →  embedding
"""

import sys
import json
import argparse
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.document_converter import DocumentConverter, PdfFormatOption
import torch

# ---------------------------------------------------------------------------
# OCR config (đã gộp từ pdf_ocr_config.py)
# ---------------------------------------------------------------------------

FORCE_OCR_DEFAULT = False
"""
Mặc định False — 7 file hiện tại đều có text layer.
Chỉ đặt True cho từng file scan cụ thể trong OCR_CONFIG bên dưới.
"""

OCR_CONFIG: dict[str, bool] = {
    # Ví dụ — bỏ comment nếu có file scan:
    # "file_scan.pdf": True,
}


def get_force_ocr(pdf_filename: str) -> bool:
    """Trả về force_ocr cho từng file PDF dựa trên OCR_CONFIG."""
    return OCR_CONFIG.get(pdf_filename, FORCE_OCR_DEFAULT)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

_use_gpu = torch.cuda.is_available()
_device_name = torch.cuda.get_device_name(0) if _use_gpu else "CPU"
print(f"[loader] 💻 Thiết bị sử dụng: {'CUDA ✅ ' + _device_name if _use_gpu else 'CPU ⚠️'}")

# ---------------------------------------------------------------------------
# Converter cache — keyed by (force_ocr, languages) tuple
# Bug fix: trước đây chỉ cache theo bool, nên cùng force_ocr khác languages
# vẫn trả về converter cũ với languages sai.
# ---------------------------------------------------------------------------

_converters: dict[tuple[bool, tuple[str, ...]], DocumentConverter] = {}


def get_converter(
    force_ocr: bool = False,
    languages: list[str] | None = None,
) -> DocumentConverter:
    """Tạo và cache DocumentConverter theo (force_ocr, languages).

    Model chỉ được load 1 lần cho mỗi cấu hình, tái sử dụng cho cả batch.
    """
    languages = languages or ["vi", "en"]
    languages = list(dict.fromkeys(languages))
    cache_key = (force_ocr, tuple(sorted(languages)))

    if cache_key in _converters:
        return _converters[cache_key]

    print(f"[loader] 🔧 Đang khởi tạo pipeline Docling (force_ocr={force_ocr}, lang={languages})...")
    t0 = time.time()

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = force_ocr
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    pipeline_options.ocr_options = EasyOcrOptions(
        lang=languages,
        force_full_page_ocr=force_ocr,
        use_gpu=_use_gpu,
    )

    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CUDA if _use_gpu else AcceleratorDevice.CPU,
        num_threads=8,
    )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    _converters[cache_key] = converter
    print(f"[loader] ✅ Pipeline sẵn sàng sau {time.time() - t0:.1f}s")
    return converter


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def load_pdf(pdf_path: str, force_ocr: bool = False, languages: list[str] | None = None) -> dict:
    """Load 1 file PDF bằng Docling, trả về raw markdown + metadata.

    Không thực hiện bất kỳ thao tác clean nào — đó là trách nhiệm của cleaner.py.
    """
    print(f"[loader] Đang convert {pdf_path} — force_ocr={force_ocr}")

    converter = get_converter(force_ocr=force_ocr, languages=languages)
    result = converter.convert(pdf_path)
    markdown = result.document.export_to_markdown()

    return {
        "markdown": markdown,
        "metadata": {
            "source_file": Path(pdf_path).name,
            "force_ocr": force_ocr,
            "languages": languages or ["vi", "en"],
            "num_pages": result.document.num_pages(),
            "status": str(result.status),
        },
    }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _process_one(pdf_path: Path, output_dir: Path, force_ocr: bool, languages: list[str]) -> None:
    """Xử lý 1 file PDF và lưu raw Markdown + metadata JSON vào output_dir."""
    result = load_pdf(str(pdf_path), force_ocr=force_ocr, languages=languages)
    md_text = result["markdown"]
    metadata = result["metadata"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw Markdown — chưa clean, dành cho cleaner.py xử lý tiếp
    md_path = output_dir / f"{pdf_path.stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    # Metadata JSON — dùng để truy ngược nguồn khi chunk → embedding → RAG
    json_path = output_dir / f"{pdf_path.stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"✅ Đã trích xuất xong:")
    print(f"   Markdown : {md_path}")
    print(f"   Metadata : {json_path}")
    print(f"   Số trang : {metadata['num_pages']}")
    print(f"   Ký tự    : {len(md_text)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF loader dùng Docling — raw extraction only")
    parser.add_argument("pdf_path", help="Đường dẫn file PDF hoặc thư mục chứa các file PDF")
    parser.add_argument(
        "--force_ocr", action="store_true",
        help="Ép OCR toàn bộ trang cho TẤT CẢ file (override OCR_CONFIG, dùng để debug)",
    )
    parser.add_argument(
        "--languages", nargs="+", default=["vi", "en"],
        help="Danh sách ngôn ngữ OCR, mặc định: vi en",
    )
    parser.add_argument(
        "--only-new", action="store_true", dest="only_new",
        help="Bỏ qua file đã có output .md trong data/interim/",
    )
    args = parser.parse_args()

    input_path = Path(args.pdf_path)
    output_dir = Path("data/interim")

    def resolve_force_ocr(pdf_file: Path) -> bool:
        """CLI --force_ocr override toàn bộ; ngược lại tra OCR_CONFIG theo filename."""
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
            if args.only_new
            and (output_dir / f"{f.stem}.md").exists()
            and (output_dir / f"{f.stem}.json").exists()
        }
        to_process = [f for f in pdf_files if f not in to_skip]
        total = len(to_process)
        processed = []

        failed = []

        for pdf_file in pdf_files:
            if pdf_file in to_skip:
                print(f"[loader] ⏭  Bỏ qua (đã có output): {pdf_file.name}")
                continue
            
            file_force_ocr = resolve_force_ocr(pdf_file)

            try:
                print(f"\n[loader] 📄 Đang xử lý ({len(processed) + 1}/{total}): {pdf_file.name} "
                      f"(force_ocr={file_force_ocr})")
                _process_one(pdf_file, output_dir, file_force_ocr, args.languages)
                processed.append(pdf_file.name)
            except Exception as e:
                failed.append({
                    "file": pdf_file.name,
                    "error": str(e)
                })
                print(f"[loader] ❌ Lỗi: {pdf_file.name}")
                print(f"         {e}")

        print("\n" + "=" * 60)
        print(f"🎉 Thành công: {len(processed)}")
        print(f"❌ Thất bại: {len(failed)}")
        print(f"⏭  Bỏ qua: {len(to_skip)}")
        if failed:
            print("\nCác file lỗi:")
            for item in failed:
                print(f"  - {item['file']}: {item['error']}")
        print("=" * 60)

    elif input_path.is_file():
        file_force_ocr = resolve_force_ocr(input_path)
        _process_one(input_path, output_dir, file_force_ocr, args.languages)

    else:
        print(f"[loader] ❌ Không tìm thấy file hoặc thư mục: {input_path}")
        sys.exit(1)
