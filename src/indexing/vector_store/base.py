from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..chunking.base import Chunk


class BaseVectorStore(ABC):
    """Interface trừu tượng cho vector store.

    Mọi implementation (Qdrant, FAISS, ChromaDB, v.v.) phải kế thừa class này
    và implement đủ 4 phương thức dưới đây.
    """

    @abstractmethod
    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Thêm danh sách chunk kèm embedding vào store.

        Args:
            chunks:     Danh sách Chunk objects (mỗi Chunk có .text và .metadata).
            embeddings: Numpy array shape (len(chunks), embedding_dim).
        """
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> List[dict]:
        """Tìm kiếm top-k chunk gần nhất với query vector.

        Args:
            query_vector: 1-D numpy array, embedding của câu hỏi.
            top_k:        Số kết quả muốn lấy.

        Returns:
            Danh sách dict, mỗi dict gồm:
              - chunk_id (str): ID duy nhất của chunk trong store.
              - text     (str): Nội dung chunk.
              - score  (float): Điểm similarity (cao hơn = liên quan hơn).
              - metadata (dict): Toàn bộ metadata gốc của chunk.
        """
        pass

    @abstractmethod
    def persist(self) -> None:
        """Lưu store xuống đĩa (no-op nếu store tự động persist)."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Tải/kết nối lại store từ đĩa."""
        pass
