from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Embed danh sách văn bản thành vector dày.

        Args:
            texts:    Danh sách chuỗi cần embed.
            is_query: True khi đây là câu hỏi (query), False khi là chunk tài
                      liệu (document). Một số model (ví dụ Qwen3-Embedding)
                      yêu cầu instruction prefix khác nhau cho 2 trường hợp.
        Returns:
            Numpy array shape (len(texts), embedding_dim).
        """
        pass
