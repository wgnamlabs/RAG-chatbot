from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Embed danh sách văn bản thành vector dày.

        Args:
            texts: Danh sách chuỗi cần embed.
            is_query: True khi đây là câu hỏi; False khi là chunk tài liệu.

        Returns:
            Numpy array shape (len(texts), embedding_dim).
        """
        raise NotImplementedError
