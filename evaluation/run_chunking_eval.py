import json
import unicodedata
from pathlib import Path
import sys

# Reconfigure stdout for Windows console to handle Vietnamese characters
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indexing.chunking import SemanticChunker, HierarchicalChunker


def _print_stats(chunker_name: str, all_chunks: list) -> None:
    """In thống kê nhanh sau mỗi chunker."""
    n = len(all_chunks)
    if not n:
        print(f"[{chunker_name}] Không có chunk nào!")
        return

    char_lens  = [len(c["text"]) for c in all_chunks]
    n_tables   = sum(1 for c in all_chunks if c["metadata"].get("is_table"))
    n_too_long = sum(1 for c in all_chunks if c["metadata"].get("too_long"))

    print(
        f"[{chunker_name}] Tổng chunks : {n}\n"
        f"             Bảng atomic : {n_tables}\n"
        f"             Quá dài     : {n_too_long}\n"
        f"             Ký tự avg   : {sum(char_lens)/n:.0f} "
        f"(min {min(char_lens)}, max {max(char_lens)})"
    )


def run_chunking(cleaned_dir: Path, output_dir: Path) -> None:
    if not cleaned_dir.exists():
        print(f"Error: {cleaned_dir} không tồn tại. Hãy chạy cleaner.py trước.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    chunkers = {
        "semantic":      SemanticChunker(),
        "hierarchical":  HierarchicalChunker(),
    }

    md_files = sorted(cleaned_dir.glob("*.md"))
    print(f"Tìm thấy {len(md_files)} file .md trong: {cleaned_dir}\n")

    for chunker_name, chunker in chunkers.items():
        print(f"\n{'='*60}")
        print(f"  {chunker_name.upper()} CHUNKER")
        print(f"{'='*60}")
        all_chunks = []

        for file_path in md_files:
            print(f"\n  📄 {file_path.name}")
            text = file_path.read_text(encoding="utf-8")

            # source là tên file .md đã clean, chuẩn hóa NFC để tránh lỗi so sánh NFD/NFC
            # trên Windows: os.fspath / Path.name có thể trả NFD khi đọc từ filesystem
            source_name = unicodedata.normalize("NFC", file_path.name)
            chunks = chunker.chunk(text, metadata={"source": source_name})

            for chunk in chunks:
                all_chunks.append({
                    "text":     chunk.text,
                    "metadata": chunk.metadata,
                })

            n_file = sum(1 for c in all_chunks if c["metadata"].get("source") == source_name)
            print(f"     → {n_file} chunks")

        out_file = output_dir / f"{chunker_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f"\n  💾 Đã lưu {len(all_chunks)} chunks → {out_file}")
        _print_stats(chunker_name, all_chunks)


if __name__ == "__main__":
    base_path   = Path(__file__).resolve().parent.parent
    cleaned_dir = base_path / "data" / "cleaned"
    output_dir  = base_path / "evaluation" / "results" / "chunks_cache"

    run_chunking(cleaned_dir, output_dir)
