from typing import List, Optional

import torch
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker as LcSemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from .base import BaseChunker, Chunk
from .config import SemanticChunkerConfig
from .table_utils import extract_tables, restore_tables_as_chunks, is_heading_only

# ---------------------------------------------------------------------------
# Regex tách câu tiếng Việt — fixed-width lookbehind (Python 3.10 re).
# Chỉ cắt khi sau dấu câu (.?!) là khoảng trắng + chữ hoa/số đầu câu mới.
# buffer_size=1 của SemanticChunker sẽ gộp lại các mảnh ngắn bị cắt nhầm.
# ---------------------------------------------------------------------------
VI_SENTENCE_SPLIT_REGEX = (
    r"(?<!\b\d\.)(?<!\b\d\d\.)(?<=[.?!])\s+(?=[A-ZĐÀÁẢÃẠĂẮẶẰẴẤẦẨẪẬÊẾỀỆỄỂÔỐỒỔỖỘƯỨỪỰỮ0-9])"
)

# Headers để phân tách section (dùng chung với HierarchicalChunker)
_HEADERS_TO_SPLIT = [
    ("#",    "Header 1"),
    ("##",   "Header 2"),
    ("###",  "Header 3"),
    ("####", "Header 4"),
]


class STEmbeddings(Embeddings):
    """Wrapper LangChain Embeddings cho SentenceTransformer."""

    def __init__(self, model_name: str, trust_remote_code: bool = False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=trust_remote_code)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()


class SemanticChunker(BaseChunker):
    """
    Chunker dựa trên ngữ nghĩa (cosine-similarity breakpoint).

    Luồng xử lý:
      1. Tách bảng ra khỏi text (bảo vệ atomic).
      2. Tách theo heading → mỗi section có breadcrumb riêng.
      3. Semantic chunking bên trong từng section.
      4. Khôi phục bảng atomic trong từng chunk đầu ra.
      5. Size guard theo token:
           - Chunk < min_chunk_tokens → gộp vào chunk trước (trừ bảng).
           - Chunk > max_chunk_tokens → THỰC SỰ split bằng
             RecursiveCharacterTextSplitter (không chỉ gắn flag).
    """

    def __init__(self, config: SemanticChunkerConfig = None):
        self.config = config or SemanticChunkerConfig()
        print(f"[SemanticChunker] Khởi tạo model {self.config.embedding_model_name}...")
        self.embeddings = STEmbeddings(self.config.embedding_model_name, trust_remote_code=self.config.trust_remote_code)

        # Tokenizer để đo token (phải khởi tạo trước overflow_splitter)
        self._tokenizer = self.embeddings.model.tokenizer

        # Splitter cho chunk vượt max_chunk_tokens — đo bằng token thật
        self._overflow_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.max_chunk_tokens,
            chunk_overlap=50,
            length_function=self._token_len,
            separators=["\n\n", "\n- ", "\n+ ", "\n", " ", ""],
        )

        # Semantic splitter (langchain experimental)
        self.chunker = LcSemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=self.config.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.config.breakpoint_threshold_amount,
            buffer_size=self.config.buffer_size,
            sentence_split_regex=VI_SENTENCE_SPLIT_REGEX,
        )

        # Header splitter để lấy breadcrumb cho từng section
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS_TO_SPLIT,
            strip_headers=False,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _semantic_split(self, text: str) -> List[str]:
        """Chạy LcSemanticChunker, fallback về nguyên đoạn nếu lỗi."""
        try:
            return [d.page_content for d in self.chunker.create_documents([text])]
        except Exception:
            return [text]

    def _split_overflow(
        self,
        piece: str,
        header_meta: dict,
        breadcrumb: str,
        base_meta: dict,
        results: List[Chunk],
        idx: int,
    ) -> int:
        """Cắt tiếp chunk quá dài, thêm vào results. Trả về idx mới."""
        for sp in self._overflow_splitter.split_text(piece):
            if not sp.strip():
                continue
            chunk_meta = base_meta.copy()
            chunk_meta.update(header_meta)
            chunk_meta["chunk_index"] = idx
            chunk_meta["is_table"]    = False
            chunk_meta["too_long"]    = True
            chunk_meta["breadcrumb"]  = breadcrumb
            
            embed_text = f"[{breadcrumb}]\n{sp}" if breadcrumb else sp
            
            results.append(Chunk(text=embed_text, metadata=chunk_meta))
            idx += 1
        return idx

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        metadata = metadata or {}

        # (1) Tách bảng khỏi text
        text_no_tables, tables = extract_tables(text)

        # (2) Tách theo heading để lấy breadcrumb metadata
        sections = self._md_splitter.split_text(text_no_tables)

        results: List[Chunk] = []
        idx = 0

        for section in sections:
            breadcrumb    = self._breadcrumb(section.metadata)
            section_text  = section.page_content

            if not section_text.strip():
                continue

            # (3) Semantic chunking bên trong section
            semantic_pieces = self._semantic_split(section_text)

            for raw_piece in semantic_pieces:
                # (4) Khôi phục bảng atomic
                for piece in restore_tables_as_chunks(raw_piece, tables):
                    if not piece.strip():
                        continue

                    is_table = piece.strip().startswith("|")
                    n_tok    = self._token_len(piece)

                    # (5a) Size guard — quá ngắn: gộp vào chunk trước
                    if (
                        not is_table
                        and n_tok < self.config.min_chunk_tokens
                        and results
                        and not results[-1].metadata.get("is_table")
                    ):
                        results[-1].text += "\n" + piece
                        continue

                    # (5b) Size guard — quá dài: THỰC SỰ split tiếp
                    if not is_table and n_tok > self.config.max_chunk_tokens:
                        idx = self._split_overflow(
                            piece, section.metadata, breadcrumb,
                            metadata, results, idx,
                        )
                        continue

                    chunk_meta = metadata.copy()
                    chunk_meta.update(section.metadata)
                    chunk_meta["chunk_index"] = idx
                    chunk_meta["is_table"]    = is_table
                    chunk_meta["too_long"]    = False
                    chunk_meta["breadcrumb"]  = breadcrumb
                    
                    embed_text = f"[{breadcrumb}]\n{piece}" if breadcrumb else piece
                    
                    results.append(Chunk(text=embed_text, metadata=chunk_meta))
                    idx += 1

        # Lọc bỏ chunk chỉ có heading/breadcrumb (semantic breakpoint rơi ngay sau heading)
        filtered: List[Chunk] = [
            chunk for chunk in results
            if chunk.metadata.get("is_table") or not is_heading_only(chunk.text)
        ]
        for i, chunk in enumerate(filtered):
            if "TBLPLACEHOLDER" in chunk.text:
                raise ValueError(f"Leaked table placeholder in chunk {i}: {chunk.text}")
            chunk.metadata["chunk_index"] = i
        return filtered
