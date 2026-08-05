from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Any


@dataclass
class RetrievalResult:
    """Kết quả trả về từ mọi retriever.

    Attributes:
        chunk_id : ID duy nhất của chunk (dạng "source_file::chunk_index").
        text     : Nội dung văn bản của chunk.
        score    : Điểm relevance (cao hơn = liên quan hơn). Ý nghĩa tùy retriever:
                   - Dense: cosine similarity ∈ [-1, 1]
                   - BM25 : BM25 score ≥ 0
                   - RRF  : 1/(rank+60) sum ∈ (0, 1]
        metadata : Toàn bộ metadata gốc từ chunk (source, chunk_index, v.v.).
        rank     : Thứ hạng trong danh sách kết quả (0-indexed, được set sau sort).
    """
    chunk_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
    rank: int = 0


class BaseRetriever(ABC):
    """Interface trừu tượng cho mọi retriever trong pipeline RAG."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Tìm kiếm top-k chunk liên quan đến query.

        Args:
            query: Câu hỏi hoặc văn bản tìm kiếm.
            top_k: Số kết quả tối đa trả về.

        Returns:
            Danh sách RetrievalResult, đã sắp xếp theo score giảm dần,
            với rank được gán từ 0 đến len-1.
        """
        pass
