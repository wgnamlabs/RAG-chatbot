"""
Embedder cho 3 model so sánh:
  - BAAI/bge-m3                     → không cần prefix
  - intfloat/multilingual-e5-base   → cần prefix "query: " / "passage: "
  - AITeamVN/Vietnamese_Embedding   → không cần prefix (fine-tune từ bge-m3)

Ghi chú về max_seq_length:
  Không ghi đè model.max_seq_length sau khi load. Một số model (ví dụ
  gte-multilingual-base) dùng custom modeling.py có bug khi max_seq_length
  bị override externally → RoPE position_ids bị corrupt. Mỗi model tự quản
  lý max_seq_length từ HuggingFace config của nó. Chunker đã giới hạn kích
  thước chunk rồi.
"""

import gc
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .base import BaseEmbedder
from .config import EmbedderConfig

# ---------------------------------------------------------------------------
# E5 prefix (intfloat/multilingual-e5-*): bắt buộc theo spec của model
# ---------------------------------------------------------------------------
_E5_QUERY_PREFIX    = "query: "
_E5_PASSAGE_PREFIX  = "passage: "


def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, config: EmbedderConfig):
        self.config = config
        self.model = None

    # ------------------------------------------------------------------
    # Load / Unload
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self.model is not None:
            return
        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[Embedder] ⚠️  CUDA không khả dụng, chuyển sang CPU.")
            device = "cpu"
        print(f"[Embedder] Loading {self.config.model_name} → {device} ...")
        self.model = SentenceTransformer(
            self.config.model_name,
            device=device,
            trust_remote_code=True,
        )
        # Không ghi đè max_seq_length sau khi load (xem docstring module)

    def unload(self) -> None:
        if self.model is None:
            return
        print(f"[Embedder] Unloading {self.config.model_name}...")
        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Tiền xử lý: thêm prefix đặc thù từng model
    # ------------------------------------------------------------------

    def _preprocess(self, texts: List[str], is_query: bool) -> List[str]:
        if _is_e5(self.config.model_name):
            prefix = _E5_QUERY_PREFIX if is_query else _E5_PASSAGE_PREFIX
            return [prefix + t for t in texts]
        # bge-m3, Vietnamese_Embedding: dùng thẳng
        return texts

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Embed danh sách văn bản.

        Args:
            texts:    Văn bản cần embed.
            is_query: True → câu hỏi; False → chunk tài liệu.
                      Ảnh hưởng đến prefix với E5 models.
        """
        if self.model is None:
            self.load()

        processed = self._preprocess(texts, is_query)
        print(f"[Embedder] Encoding {len(texts)} texts "
              f"(is_query={is_query}, batch_size={self.config.batch_size})...")
        return self.model.encode(
            processed,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
