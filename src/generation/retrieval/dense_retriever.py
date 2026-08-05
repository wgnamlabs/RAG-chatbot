"""
DenseRetriever — Wrap QdrantVectorStore + SentenceTransformerEmbedder thành BaseRetriever.

Tổ hợp mặc định: AITeamVN/Vietnamese_Embedding + Qdrant local store (cosine).
"""

from pathlib import Path
from typing import List, Optional

from .base import BaseRetriever, RetrievalResult


class DenseRetriever(BaseRetriever):
    """Retriever dùng dense vector (cosine similarity qua Qdrant local store).

    Args:
        store:    QdrantVectorStore đã được load (hoặc sẽ tự load khi retrieve).
        embedder: SentenceTransformerEmbedder đã được load.

    Ví dụ:
        store = QdrantVectorStore(config)
        store.load()
        embedder = SentenceTransformerEmbedder(emb_config)
        embedder.load()
        retriever = DenseRetriever(store=store, embedder=embedder)
        results = retriever.retrieve("đái tháo đường thai kỳ", top_k=10)
    """

    def __init__(self, store, embedder):
        self._store   = store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Embed query → search Qdrant → trả về RetrievalResult.

        Args:
            query: Câu hỏi (embed với is_query=True).
            top_k: Số kết quả tối đa.

        Returns:
            Danh sách RetrievalResult, sắp xếp theo cosine similarity giảm dần.
        """
        import numpy as np

        # Embed câu hỏi (is_query=True để dùng đúng prefix nếu model yêu cầu)
        query_emb = self._embedder.encode([query], is_query=True)[0]

        # Tìm kiếm trong Qdrant
        hits = self._store.search(query_vector=query_emb, top_k=top_k)

        results = []
        for rank, hit in enumerate(hits):
            results.append(RetrievalResult(
                chunk_id=hit["chunk_id"],
                text=hit["text"],
                score=hit["score"],
                metadata=hit["metadata"],
                rank=rank,
            ))
        return results
