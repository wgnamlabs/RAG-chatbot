import argparse
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

# Reconfigure stdout for Windows console to handle Vietnamese characters.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src is in path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indexing.chunking import SemanticChunker, HierarchicalChunker
from indexing.chunking.config import SemanticChunkerConfig, HierarchicalChunkerConfig


DEFAULT_MODEL = "AITeamVN/Vietnamese_Embedding_v2"
DEFAULT_MAX_TOKENS = 1024


def _percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _stats(chunker_name: str, all_chunks: list) -> dict:
    n = len(all_chunks)
    if not n:
        return {
            "chunker": chunker_name,
            "total_chunks": 0,
        }

    token_lens = [int(c["metadata"].get("token_count", 0)) for c in all_chunks]
    char_lens = [len(c["text"]) for c in all_chunks]

    table_chunks = [c for c in all_chunks if c["metadata"].get("is_table")]
    too_long = [c for c in all_chunks if c["metadata"].get("too_long")]
    overflow = [c for c in all_chunks if c["metadata"].get("overflow_split")]
    split_chunks = [c for c in all_chunks if c["metadata"].get("was_split")]
    semantic_fallback = [c for c in all_chunks if c["metadata"].get("semantic_fallback")]
    table_split_parts = [
        c for c in table_chunks
        if int(c["metadata"].get("table_parts", 1)) > 1
    ]

    # QA cho 3 lỗi structural quan trọng của corpus này.
    chunk_ids = [str(c["metadata"].get("chunk_id", "")) for c in all_chunks]
    duplicate_chunk_ids = len(chunk_ids) - len(set(chunk_ids))

    def content_without_breadcrumb(text: str) -> str:
        lines = text.splitlines()
        if lines and lines[0].startswith("[") and lines[0].endswith("]"):
            lines = lines[1:]
        return "\n".join(lines).lstrip("\n")

    leading_punctuation_chunks = 0
    nested_list_lines = 0
    for c in all_chunks:
        body = content_without_breadcrumb(c["text"])
        if body.lstrip().startswith((".", ",", ";", ":", "?", "!")):
            leading_punctuation_chunks += 1
        nested_list_lines += sum(
            1
            for line in body.splitlines()
            if re.match(r"^[ \t]{2,}[-+*]\s+", line)
        )

    return {
        "chunker": chunker_name,
        "total_chunks": n,
        "table_chunks": len(table_chunks),
        "table_split_parts": len(table_split_parts),
        "was_split_chunks": len(split_chunks),
        "overflow_split_chunks": len(overflow),
        "actual_too_long_chunks": len(too_long),
        "semantic_fallback_chunks": len(semantic_fallback),
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "leading_punctuation_chunks": leading_punctuation_chunks,
        "nested_list_lines_preserved": nested_list_lines,
        "tokens": {
            "avg": round(statistics.mean(token_lens), 2),
            "median": round(statistics.median(token_lens), 2),
            "p95": round(_percentile(token_lens, 0.95), 2),
            "min": min(token_lens),
            "max": max(token_lens),
        },
        "chars": {
            "avg": round(statistics.mean(char_lens), 2),
            "median": round(statistics.median(char_lens), 2),
            "p95": round(_percentile(char_lens, 0.95), 2),
            "min": min(char_lens),
            "max": max(char_lens),
        },
    }


def _print_stats(stats: dict) -> None:
    if not stats.get("total_chunks"):
        print(f"[{stats['chunker']}] Không có chunk nào!")
        return

    t = stats["tokens"]
    print(
        f"[{stats['chunker']}] Tổng chunks        : {stats['total_chunks']}\n"
        f"             Bảng chunks        : {stats['table_chunks']}\n"
        f"             Bảng split parts   : {stats['table_split_parts']}\n"
        f"             was_split          : {stats['was_split_chunks']}\n"
        f"             overflow_split     : {stats['overflow_split_chunks']}\n"
        f"             ACTUAL too_long    : {stats['actual_too_long_chunks']}\n"
        f"             semantic fallback  : {stats['semantic_fallback_chunks']}\n"
        f"             duplicate chunk_id : {stats['duplicate_chunk_ids']}\n"
        f"             bad boundaries     : {stats['leading_punctuation_chunks']}\n"
        f"             nested list lines  : {stats['nested_list_lines_preserved']}\n"
        f"             Tokens avg/med/p95 : {t['avg']} / {t['median']} / {t['p95']}\n"
        f"             Tokens min/max     : {t['min']} / {t['max']}"
    )


def build_chunkers(model_name: str, max_tokens: int) -> dict:
    """Hai method dùng cùng tokenizer/model family và cùng max token budget."""
    semantic_cfg = SemanticChunkerConfig(
        embedding_model_name=model_name,
        max_chunk_tokens=max_tokens,
        raise_on_semantic_error=True,
    )
    hierarchical_cfg = HierarchicalChunkerConfig(
        tokenizer_model_name=model_name,
        child_chunk_size=max_tokens,
        child_chunk_overlap=min(100, max_tokens // 5),
    )

    return {
        "semantic": SemanticChunker(semantic_cfg),
        "hierarchical": HierarchicalChunker(hierarchical_cfg),
    }


def run_chunking(
    cleaned_dir: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> None:
    if not cleaned_dir.exists():
        raise FileNotFoundError(
            f"{cleaned_dir} không tồn tại. Hãy đặt các *_clean.md vào thư mục này."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Chỉ lấy corpus đã clean để tránh vô tình chunk cả MD thô/intermediate.
    md_files = sorted(cleaned_dir.glob("*_clean.md"))
    if not md_files:
        raise FileNotFoundError(f"Không tìm thấy file *_clean.md trong {cleaned_dir}")

    print(f"Tìm thấy {len(md_files)} file clean trong: {cleaned_dir}")
    print(f"Token budget chung: {max_tokens}")
    print(f"Tokenizer/model chung: {model_name}\n")

    chunkers = build_chunkers(model_name, max_tokens)
    summary = {
        "experiment": {
            "cleaned_dir": str(cleaned_dir),
            "files": [p.name for p in md_files],
            "tokenizer_model": model_name,
            "max_chunk_tokens": max_tokens,
            "note": (
                "Đây là structural chunking evaluation. Retrieval metrics như "
                "Recall@k/MRR/nDCG cần chạy ở bước retrieval với cùng query set."
            ),
        },
        "methods": {},
    }

    for chunker_name, chunker in chunkers.items():
        print(f"\n{'=' * 64}")
        print(f"  {chunker_name.upper()} CHUNKER")
        print(f"{'=' * 64}")

        all_chunks = []
        per_file = {}

        for file_path in md_files:
            print(f"\n  📄 {file_path.name}")
            text = file_path.read_text(encoding="utf-8")
            source_name = unicodedata.normalize("NFC", file_path.name)

            chunks = chunker.chunk(text, metadata={"source": source_name})
            serialized = [
                {"text": chunk.text, "metadata": chunk.metadata}
                for chunk in chunks
            ]
            all_chunks.extend(serialized)
            per_file[source_name] = _stats(chunker_name, serialized)
            print(f"     → {len(serialized)} chunks")

        out_file = output_dir / f"{chunker_name}.json"
        out_file.write_text(
            json.dumps(all_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        method_stats = _stats(chunker_name, all_chunks)
        method_stats["per_file"] = per_file
        summary["methods"][chunker_name] = method_stats

        print(f"\n  💾 Đã lưu {len(all_chunks)} chunks → {out_file}")
        _print_stats(method_stats)

        # Fail-fast QA: các lỗi này làm experiment/index không còn sạch.
        if method_stats.get("actual_too_long_chunks", 0) > 0:
            raise RuntimeError(
                f"{chunker_name}: có {method_stats['actual_too_long_chunks']} chunk "
                f"vượt {max_tokens} tokens."
            )
        if method_stats.get("duplicate_chunk_ids", 0) > 0:
            raise RuntimeError(
                f"{chunker_name}: có {method_stats['duplicate_chunk_ids']} chunk_id bị trùng."
            )
        if method_stats.get("leading_punctuation_chunks", 0) > 0:
            raise RuntimeError(
                f"{chunker_name}: còn {method_stats['leading_punctuation_chunks']} chunk "
                "mở đầu bằng dấu câu; kiểm tra overflow boundary."
            )

    summary_file = output_dir / "chunking_summary.json"
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n📊 Summary → {summary_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="So sánh Structure-aware Semantic vs Hierarchical Chunking"
    )
    base_path = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--cleaned-dir",
        type=Path,
        default=base_path / "data" / "cleaned",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_path / "evaluation" / "results" / "chunks_cache",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_chunking(
        cleaned_dir=args.cleaned_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        max_tokens=args.max_tokens,
    )
