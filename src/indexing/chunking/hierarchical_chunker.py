from typing import List, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .base import BaseChunker, Chunk
from .config import HierarchicalChunkerConfig
from .table_utils import extract_tables, restore_tables_as_chunks, is_heading_only


class HierarchicalChunker(BaseChunker):
    """
    Chunker 2 tầng:
      Tầng 1 — MarkdownHeaderTextSplitter: tách theo heading → section có breadcrumb.
      Tầng 2 — RecursiveCharacterTextSplitter: cắt tiếp section dài → child chunks.

    Cải tiến:
      • Bảo vệ bảng atomic (placeholder ASCII, không phải NUL).
      • Breadcrumb nhúng vào đầu mọi chunk (kể cả chunk 2, 3... của section dài).
      • length_function có thể là tokenizer để đo bằng token thay vì ký tự.
      • Lọc bỏ chunk chỉ có heading/breadcrumb, không có nội dung thật.
    """

    def __init__(self, config: HierarchicalChunkerConfig = None):
        self.config = config or HierarchicalChunkerConfig()

        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.config.headers_to_split_on,
            strip_headers=False,
        )
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.child_chunk_size,
            chunk_overlap=self.config.child_chunk_overlap,
            length_function=self.config.length_function,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _breadcrumb(meta: dict) -> str:
        levels = [
            meta[k]
            for k in sorted(meta)
            if k.startswith("Header") and meta.get(k)
        ]
        return " > ".join(levels)

    # is_heading_only dùng chung với SemanticChunker — định nghĩa trong table_utils.py

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        metadata = metadata or {}

        # (1) Tách bảng ra khỏi text trước khi split
        text_no_tables, tables = extract_tables(text)

        # (2) Tách theo heading
        md_docs = self.md_splitter.split_text(text_no_tables)

        raw_results: List[Chunk] = []
        global_chunk_idx = 0

        for parent_doc in md_docs:
            breadcrumb = self._breadcrumb(parent_doc.metadata)

            # (3) Cắt tiếp nội dung dài trong mỗi section
            child_docs = self.char_splitter.split_documents([parent_doc])
            
            was_split = len(child_docs) > 1

            for child_doc in child_docs:
                # (4) Khôi phục bảng: mỗi bảng → chunk atomic riêng
                for piece in restore_tables_as_chunks(child_doc.page_content, tables):
                    if not piece.strip():
                        continue

                    is_table = piece.strip().startswith("|")

                    # (5) Nhúng breadcrumb vào đầu text sẽ được embed
                    embed_text = f"[{breadcrumb}]\n{piece}" if breadcrumb else piece

                    chunk_meta = metadata.copy()
                    chunk_meta.update(child_doc.metadata)
                    chunk_meta["chunk_index"] = global_chunk_idx
                    chunk_meta["breadcrumb"]  = breadcrumb
                    chunk_meta["is_table"]    = is_table
                    chunk_meta["too_long"]    = was_split

                    raw_results.append(Chunk(text=embed_text, metadata=chunk_meta))
                    global_chunk_idx += 1

        # (6) Lọc bỏ chunk chỉ có heading/breadcrumb, không có nội dung thật
        filtered: List[Chunk] = [
            chunk for chunk in raw_results
            if chunk.metadata.get("is_table") or not is_heading_only(chunk.text)
        ]

        # Đánh lại chunk_index sau khi lọc
        for i, chunk in enumerate(filtered):
            if "TBLPLACEHOLDER" in chunk.text:
                raise ValueError(f"Leaked table placeholder in chunk {i}: {chunk.text}")
            chunk.metadata["chunk_index"] = i

        return filtered
