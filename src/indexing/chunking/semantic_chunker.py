from typing import List, Optional, Tuple

import torch
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker as LcSemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from .base import BaseChunker, Chunk
from .config import SemanticChunkerConfig
from .markdown_utils import split_markdown_sections
from .table_utils import (
    extract_tables,
    is_heading_only,
    is_table_chunk,
    restore_tables_as_chunks,
    split_table_by_rows,
    table_caption,
)


# Fixed-width lookbehind, phù hợp Python 3.10+.
# Tránh cắt sau các số thứ tự 1. / 12.; chỉ cắt khi sau .?! là đầu câu mới.
VI_SENTENCE_SPLIT_REGEX = (
    r"(?<!\b\d\.)(?<!\b\d\d\.)(?<=[.?!])\s+"
    r"(?=[A-ZĐÀÁẢÃẠĂẮẶẰẴẤẦẨẪẬÊẾỀỆỄỂÔỐỒỔỖỘƯỨỪỰỮ0-9])"
)


class STEmbeddings(Embeddings):
    """Wrapper LangChain cho SentenceTransformer."""

    def __init__(self, model_name: str, trust_remote_code: bool = False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()


class SemanticChunker(BaseChunker):
    """Structure-aware semantic chunker.

    1. Bảo vệ bảng.
    2. Split Markdown H1-H6 bằng parser giữ nguyên whitespace/list nesting.
    3. Semantic breakpoint bên trong từng section.
    4. Size guard bằng token thật.
    5. Chunk ngắn chỉ được merge với chunk trước nếu CÙNG breadcrumb.
    6. Bảng quá dài được chia theo row, không cắt ngang ô/hàng.
    7. Overflow splitter giữ dấu câu ở CUỐI chunk trước.
    """

    def __init__(self, config: SemanticChunkerConfig = None):
        self.config = config or SemanticChunkerConfig()
        print(f"[SemanticChunker] Khởi tạo model {self.config.embedding_model_name}...")

        self.embeddings = STEmbeddings(
            self.config.embedding_model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._tokenizer = self.embeddings.model.tokenizer
        self.semantic_fallback_count = 0

        self.chunker = LcSemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=self.config.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.config.breakpoint_threshold_amount,
            buffer_size=self.config.buffer_size,
            sentence_split_regex=VI_SENTENCE_SPLIT_REGEX,
        )

    def _token_len(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

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

    @staticmethod
    def _strip_breadcrumb(embed_text: str, breadcrumb: str) -> str:
        prefix = f"[{breadcrumb}]\n" if breadcrumb else ""
        if prefix and embed_text.startswith(prefix):
            return embed_text[len(prefix):]
        return embed_text

    def _content_budget(self, breadcrumb: str) -> int:
        overhead = self._token_len(f"[{breadcrumb}]\n") if breadcrumb else 0
        budget = self.config.max_chunk_tokens - overhead
        if budget < 32:
            raise ValueError(
                f"Breadcrumb quá dài ({overhead} tokens), không còn đủ budget "
                f"cho content trong max={self.config.max_chunk_tokens}."
            )
        return budget

    def _semantic_split(self, text: str) -> Tuple[List[str], bool]:
        """Return (pieces, used_fallback). Evaluation mặc định raise khi lỗi."""
        try:
            return [d.page_content for d in self.chunker.create_documents([text])], False
        except Exception:
            if self.config.raise_on_semantic_error:
                raise
            self.semantic_fallback_count += 1
            return [text], True

    def _overflow_splitter(self, content_budget: int) -> RecursiveCharacterTextSplitter:
        overlap = min(self.config.overflow_overlap_tokens, max(0, content_budget - 1))
        return RecursiveCharacterTextSplitter(
            chunk_size=content_budget,
            chunk_overlap=overlap,
            length_function=self._token_len,
            # Giữ punctuation ở cuối chunk trước. Không dùng "\n- " separator
            # vì bullet marker phải thuộc chunk sau.
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
        idx: int,
        *,
        is_table: bool = False,
        semantic_split: bool = False,
        semantic_fallback: bool = False,
        overflow_split: bool = False,
        split_reason: Optional[str] = None,
        table_part: Optional[int] = None,
        table_parts: Optional[int] = None,
    ) -> Chunk:
        embed_text = self._embed_text(piece, breadcrumb)
        token_count = self._token_len(embed_text)

        meta = base_meta.copy()
        meta.update(header_meta)
        meta.update(
            {
                "chunk_index": idx,
                "breadcrumb": breadcrumb,
                "is_table": is_table,
                "content_token_count": self._token_len(piece),
                "token_count": token_count,
                "semantic_split": semantic_split,
                "semantic_fallback": semantic_fallback,
                "overflow_split": overflow_split,
                "was_split": semantic_split or overflow_split,
                "split_reason": split_reason,
                "too_long": token_count > self.config.max_chunk_tokens,
            }
        )

        if is_table:
            meta["table_caption"] = table_caption(piece)
            meta["table_part"] = table_part or 1
            meta["table_parts"] = table_parts or 1
            if (table_parts or 1) > 1:
                meta["was_split"] = True
                meta["split_reason"] = "table_rows"

        return Chunk(text=embed_text, metadata=meta)

    def _split_overflow(
        self,
        piece: str,
        header_meta: dict,
        breadcrumb: str,
        base_meta: dict,
        results: List[Chunk],
        idx: int,
        *,
        semantic_split: bool,
        semantic_fallback: bool,
        content_budget: int,
    ) -> int:
        splitter = self._overflow_splitter(content_budget)
        for sp in splitter.split_text(piece):
            # Preserve leading spaces of a nested bullet at a chunk boundary.
            sp = sp.strip("\n")
            if not sp.strip():
                continue
            results.append(
                self._make_chunk(
                    sp,
                    breadcrumb,
                    header_meta,
                    base_meta,
                    idx,
                    semantic_split=semantic_split,
                    semantic_fallback=semantic_fallback,
                    overflow_split=True,
                    split_reason="overflow_token_guard",
                )
            )
            idx += 1
        return idx

    def _try_merge_short_piece(
        self,
        piece: str,
        breadcrumb: str,
        results: List[Chunk],
    ) -> bool:
        """Merge chunk ngắn chỉ khi cùng section và không vượt max token budget."""
        if not results:
            return False

        prev = results[-1]
        if prev.metadata.get("is_table"):
            return False
        if prev.metadata.get("breadcrumb") != breadcrumb:
            return False

        prev_content = self._strip_breadcrumb(prev.text, breadcrumb)
        merged_content = f"{prev_content}\n{piece}".strip("\n")
        merged_embed = self._embed_text(merged_content, breadcrumb)

        if self._token_len(merged_embed) > self.config.max_chunk_tokens:
            return False

        prev.text = merged_embed
        prev.metadata["content_token_count"] = self._token_len(merged_content)
        prev.metadata["token_count"] = self._token_len(merged_embed)
        prev.metadata["merged_short_piece"] = True
        prev.metadata["too_long"] = False
        return True

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        metadata = metadata or {}

        text_no_tables, tables = extract_tables(text)
        # Không dùng MarkdownHeaderTextSplitter vì nó strip leading whitespace
        # và làm nested bullets mất quan hệ cha/con.
        sections = split_markdown_sections(
            text_no_tables,
            self.config.headers_to_split_on,
        )

        results: List[Chunk] = []
        idx = 0

        for section in sections:
            breadcrumb = self._breadcrumb(section.metadata)
            section_text = section.page_content.strip("\n")
            if not section_text.strip():
                continue

            content_budget = self._content_budget(breadcrumb)
            semantic_pieces, used_fallback = self._semantic_split(section_text)
            section_semantically_split = len(semantic_pieces) > 1

            for raw_piece in semantic_pieces:
                for piece in restore_tables_as_chunks(raw_piece, tables):
                    # Do not .strip(): it removes indentation from nested bullets.
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
                                    semantic_split=section_semantically_split,
                                    semantic_fallback=used_fallback,
                                    split_reason="table_rows" if len(table_parts) > 1 else None,
                                    table_part=part_no,
                                    table_parts=len(table_parts),
                                )
                            )
                            idx += 1
                        continue

                    piece_tokens = self._token_len(piece)

                    if piece_tokens < self.config.min_chunk_tokens:
                        if self._try_merge_short_piece(piece, breadcrumb, results):
                            continue

                    if piece_tokens > content_budget:
                        idx = self._split_overflow(
                            piece,
                            section.metadata,
                            breadcrumb,
                            metadata,
                            results,
                            idx,
                            semantic_split=section_semantically_split,
                            semantic_fallback=used_fallback,
                            content_budget=content_budget,
                        )
                        continue

                    results.append(
                        self._make_chunk(
                            piece,
                            breadcrumb,
                            section.metadata,
                            metadata,
                            idx,
                            semantic_split=section_semantically_split,
                            semantic_fallback=used_fallback,
                            split_reason="semantic_breakpoint" if section_semantically_split else None,
                        )
                    )
                    idx += 1

        filtered = [
            chunk
            for chunk in results
            if chunk.metadata.get("is_table") or not is_heading_only(chunk.text)
        ]

        # Re-index after filtering and create globally safe vector/document IDs.
        for new_idx, chunk in enumerate(filtered):
            if "TBLPLACEHOLDER" in chunk.text:
                raise ValueError(f"Leaked table placeholder in chunk {new_idx}: {chunk.text}")
            chunk.metadata["chunk_index"] = new_idx
            source = str(chunk.metadata.get("source") or "unknown_source")
            chunk.metadata["chunk_id"] = f"{source}::{new_idx}"

        return filtered
