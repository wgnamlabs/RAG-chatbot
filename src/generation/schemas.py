"""
schemas.py — Pydantic v2 models dùng xuyên suốt generation pipeline.

QUAN TRỌNG: Chunk ở đây KHÁC với indexing.chunking.base.Chunk (là dataclass).
  - indexing.chunking.base.Chunk: dùng khi chunking/indexing PDF → vector store
  - generation.schemas.Chunk   : dùng trong retrieval → rerank → prompt → output

Tài liệu là file .md, KHÔNG có số trang → dùng `section` (breadcrumb/header)
thay cho `page` để citation có ý nghĩa.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Đơn vị văn bản được retrieve từ Qdrant/BM25, dùng qua toàn bộ pipeline.

    Mapping từ Qdrant payload:
        chunk_id  ← payload["chunk_id"]       (format "source_file::chunk_index")
        text      ← payload["text"]
        source    ← payload["source_file"]    (tên file .md)
        section   ← payload["breadcrumb"]     (fallback: payload["Header 2"])
        score     ← cập nhật theo từng bước (retrieval → rerank)

    Không có field `page` vì tài liệu là Markdown, không có số trang.
    """

    chunk_id: str = Field(..., description="ID duy nhất, format 'source_file::chunk_index'")
    text: str = Field(..., description="Nội dung chunk")
    source: str = Field(..., description="Tên file .md nguồn")
    section: str | None = Field(
        default=None,
        description="Tên mục/section từ breadcrumb hoặc Header 2 trong payload Qdrant",
    )
    score: float = Field(..., description="Điểm từ bước retrieval/rerank gần nhất")

    def citation(self) -> str:
        """Chuỗi trích dẫn ngắn để dùng trong câu trả lời."""
        sec = f", mục: {self.section}" if self.section else ""
        return f"(Nguồn: {self.source}{sec})"


class RetrievalDebugInfo(BaseModel):
    """Debug info theo dõi rank của chunk qua từng bước retrieval.

    Dùng khi cần trace tại sao một chunk được chọn (dense vs sparse vs rerank).
    """

    dense_rank: int | None = Field(default=None, description="Rank trong dense search (1-indexed)")
    sparse_rank: int | None = Field(default=None, description="Rank trong BM25 search (1-indexed)")
    rrf_score: float | None = Field(default=None, description="Điểm RRF sau khi kết hợp")
    rerank_score: float | None = Field(default=None, description="Điểm CrossEncoder reranker")


class PipelineOutput(BaseModel):
    """Output đầy đủ của run_pipeline(), bao gồm câu trả lời và debug info."""

    original_query: str = Field(..., description="Câu hỏi gốc của người dùng")
    rewritten_query: str = Field(..., description="Câu hỏi sau khi rewrite (hoặc = original nếu fallback)")
    answer: str = Field(..., description="Câu trả lời từ LLM hoặc thông báo lỗi/từ chối")
    sources_used: list[Chunk] = Field(
        default_factory=list,
        description="Các chunk được đưa vào prompt, theo thứ tự sandwich",
    )
    latency_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Latency từng bước: {'rewrite': 0.2, 'retrieval': 0.5, 'rerank': 0.3, 'generation': 1.1}",
    )

    @property
    def total_latency(self) -> float:
        return sum(self.latency_breakdown.values())
