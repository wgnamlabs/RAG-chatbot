from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from .base import BaseChunker, Chunk
from .config import HierarchicalChunkerConfig
from .markdown_utils import split_markdown_sections
from .table_utils import (
    extract_tables,
    is_heading_only,
    is_table_chunk,
    restore_tables_as_chunks,
    split_table_by_rows,
    table_caption,
)


class HierarchicalChunker(BaseChunker):
    """Structure-aware hierarchical chunker.

    Tầng 1: split Markdown H1-H6 bằng parser giữ nguyên whitespace/list nesting.
    Tầng 2: RecursiveCharacterTextSplitter bên trong từng section, đo bằng token.

    Bảng được bảo vệ atomic; bảng vượt budget được chia theo row và lặp header.
    """

    def __init__(self, config: HierarchicalChunkerConfig = None):
        self.config = config or HierarchicalChunkerConfig()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_model_name,
            trust_remote_code=self.config.trust_remote_code,
        )

    def _token_len(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _split_length(self, text: str) -> int:
        if self.config.length_function is not None:
            return self.config.length_function(text)
        return self._token_len(text)

    @staticmethod
    def _breadcrumb(meta: dict) -> str:
        levels = [
            meta[k]
            for k in sorted(meta)
            if k.startswith("Header") and meta.get(k)
        ]
        return " > ".join(levels)

    @staticmethod
    def _embed_text(piece: str, breadcrumb: str) -> str:
        return f"[{breadcrumb}]\n{piece}" if breadcrumb else piece

    def _content_budget(self, breadcrumb: str) -> int:
        overhead = self._token_len(f"[{breadcrumb}]\n") if breadcrumb else 0
        budget = self.config.child_chunk_size - overhead
        if budget < 32:
            raise ValueError(
                f"Breadcrumb quá dài ({overhead} tokens), không còn đủ budget "
                f"cho content trong max={self.config.child_chunk_size}."
            )
        return budget

    def _make_splitter(self, content_budget: int) -> RecursiveCharacterTextSplitter:
        overlap = min(self.config.child_chunk_overlap, max(0, content_budget - 1))
        return RecursiveCharacterTextSplitter(
            chunk_size=content_budget,
            chunk_overlap=overlap,
            length_function=self._split_length,
            # Không dùng separator kiểu "\n- " khi keep_separator="end", vì
            # marker bullet phải thuộc chunk sau. Split ở newline là đủ và giữ
            # nguyên nested-list indentation.
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
            keep_separator="end",
            strip_whitespace=False,
        )

    def _make_chunk(
        self,
        piece: str,
        breadcrumb: str,
        header_meta: dict,
        base_meta: dict,
        chunk_index: int,
        *,
        is_table: bool,
        was_split: bool,
        split_reason: Optional[str],
        table_part: Optional[int] = None,
        table_parts: Optional[int] = None,
    ) -> Chunk:
        embed_text = self._embed_text(piece, breadcrumb)
        token_count = self._token_len(embed_text)

        meta = base_meta.copy()
        meta.update(header_meta)
        meta.update(
            {
                "chunk_index": chunk_index,
                "breadcrumb": breadcrumb,
                "is_table": is_table,
                "content_token_count": self._token_len(piece),
                "token_count": token_count,
                "was_split": was_split,
                "overflow_split": False,
                "split_reason": split_reason,
                "too_long": token_count > self.config.child_chunk_size,
            }
        )

        if is_table:
            meta["table_caption"] = table_caption(piece)
            meta["table_part"] = table_part or 1
            meta["table_parts"] = table_parts or 1

        return Chunk(text=embed_text, metadata=meta)

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        metadata = metadata or {}

        text_no_tables, tables = extract_tables(text)
        # Custom splitter giữ nguyên leading spaces của nested bullets.
        sections = split_markdown_sections(
            text_no_tables,
            self.config.headers_to_split_on,
        )

        results: List[Chunk] = []
        idx = 0

        for section in sections:
            breadcrumb = self._breadcrumb(section.metadata)
            content_budget = self._content_budget(breadcrumb)
            splitter = self._make_splitter(content_budget)

            child_texts = splitter.split_text(section.page_content)
            section_was_split = len(child_texts) > 1

            for child_text in child_texts:
                for piece in restore_tables_as_chunks(child_text, tables):
                    # Chỉ bỏ newline biên; .strip() sẽ phá indent của nested bullet
                    # nếu chunk bắt đầu đúng ở một child bullet.
                    piece = piece.strip("\n")
                    if not piece.strip():
                        continue

                    if is_table_chunk(piece):
                        table_parts = split_table_by_rows(
                            piece,
                            max_tokens=content_budget,
                            token_len=self._token_len,
                            overlap_rows=self.config.table_row_overlap,
                        )
                        for part_no, table_part_text in enumerate(table_parts, start=1):
                            results.append(
                                self._make_chunk(
                                    table_part_text,
                                    breadcrumb,
                                    section.metadata,
                                    metadata,
                                    idx,
                                    is_table=True,
                                    was_split=section_was_split or len(table_parts) > 1,
                                    split_reason="table_rows" if len(table_parts) > 1 else None,
                                    table_part=part_no,
                                    table_parts=len(table_parts),
                                )
                            )
                            idx += 1
                        continue

                    chunk = self._make_chunk(
                        piece,
                        breadcrumb,
                        section.metadata,
                        metadata,
                        idx,
                        is_table=False,
                        was_split=section_was_split,
                        split_reason="section_recursive" if section_was_split else None,
                    )
                    if not is_heading_only(chunk.text):
                        results.append(chunk)
                        idx += 1

        # Re-index after filtering and create globally safe vector/document IDs.
        for new_idx, chunk in enumerate(results):
            if "TBLPLACEHOLDER" in chunk.text:
                raise ValueError(f"Leaked table placeholder in chunk {new_idx}: {chunk.text}")
            chunk.metadata["chunk_index"] = new_idx
            source = str(chunk.metadata.get("source") or "unknown_source")
            chunk.metadata["chunk_id"] = f"{source}::{new_idx}"

        return results
