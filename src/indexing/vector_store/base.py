from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..chunking.base import Chunk


class BaseVectorStore(ABC):
    """Interface chung cho vector store của RAG pipeline."""

    @abstractmethod
    def add(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        **kwargs,
    ) -> None:
        """Thêm chunks và embeddings vào vector store."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 15,
    ) -> List[dict]:
        """Tìm top-k chunks gần query vector nhất."""
        raise NotImplementedError

    @abstractmethod
    def persist(self) -> None:
        """Persist store nếu backend yêu cầu."""
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        """Khởi tạo hoặc kết nối vector store."""
        raise NotImplementedError
