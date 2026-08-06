"""
Embedder cho các model so sánh:
  - BAAI/bge-m3                      → không cần prefix
  - intfloat/multilingual-e5-*       → cần prefix "query: " / "passage: "
  - AITeamVN/Vietnamese_Embedding*   → không cần prefix (fine-tune từ bge-m3)
  - Qwen/Qwen3-Embedding-4B          → không cần prefix, model nặng, dùng auto-batch
  - nvidia/Nemotron-3-Embed-8B-BF16  → model nặng nhất, luôn dùng auto-batch

Ghi chú về max_seq_length:
  Không ghi đè model.max_seq_length sau khi load. Một số model (ví dụ
  gte-multilingual-base) dùng custom modeling.py có bug khi max_seq_length
  bị override externally → RoPE position_ids bị corrupt. Mỗi model tự quản
  lý max_seq_length từ HuggingFace config của nó. Chunker đã giới hạn kích
  thước chunk rồi.

Ghi chú về auto_batch:
  Với các model nặng (Qwen3-Embedding-4B, Nemotron-3-Embed-8B), VRAM T4
  (16GB) khá sát giới hạn và độ dài chunk trong corpus không đều nhau, nên
  encode() sẽ TỰ ĐỘNG lùi batch_size (chia đôi) mỗi khi gặp CUDA OOM, thay
  vì crash và mất toàn bộ tiến trình đã chạy. Bật qua config.auto_batch.
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
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


def _log_gpu_memory(tag: str = "") -> None:
    """In VRAM đã cấp phát / tổng VRAM, để theo dõi thực tế thay vì đoán
    qua thanh RAM GPU trên UI Colab (không phải lúc nào cũng real-time)."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    prefix = f"[{tag}] " if tag else ""
    print(f"{prefix}GPU memory: {allocated:.2f}GB / {total:.2f}GB")


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
        _log_gpu_memory(f"Loaded {self.config.model_name}")

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
        # bge-m3, Vietnamese_Embedding, Qwen3-Embedding, Nemotron: dùng thẳng
        return texts

    # ------------------------------------------------------------------
    # Encode (có auto-batch fallback khi OOM)
    # ------------------------------------------------------------------

    def _encode_with_auto_batch(self, processed: List[str]) -> np.ndarray:
        """Thử encode với batch_size ban đầu, tự động chia đôi batch_size
        mỗi khi gặp CUDA OOM cho tới khi thành công hoặc chạm min_batch_size.
        Dùng cho các model nặng (Qwen3-Embedding-4B, Nemotron-3-Embed-8B)."""
        batch_size = self.config.batch_size
        min_batch_size = self.config.min_batch_size

        while batch_size >= min_batch_size:
            try:
                result = self.model.encode(
                    processed,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                )
                _log_gpu_memory(f"After encode (batch_size={batch_size})")
                return result
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                new_batch_size = batch_size // 2
                print(
                    f"[Embedder][OOM] batch_size={batch_size} vượt VRAM, "
                    f"giảm xuống {new_batch_size} và thử lại..."
                )
                batch_size = new_batch_size

        raise RuntimeError(
            f"[Embedder] Không thể encode dù đã giảm batch_size xuống "
            f"{min_batch_size} (model={self.config.model_name}). "
            f"Cân nhắc giảm max_seq_length hoặc dùng model nhẹ hơn."
        )

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
        print(
            f"[Embedder] Encoding {len(texts)} texts "
            f"(is_query={is_query}, batch_size={self.config.batch_size}, "
            f"auto_batch={self.config.auto_batch})..."
        )

        if self.config.auto_batch and self.config.device == "cuda":
            return self._encode_with_auto_batch(processed)

        # Model nhẹ (AITeamVN, bge-m3, e5-large...): giữ nguyên đường cũ,
        # không cần overhead try/except vì hiếm khi OOM.
        return self.model.encode(
            processed,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )