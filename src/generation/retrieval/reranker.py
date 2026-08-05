"""
CrossEncoderReranker — Reranker dùng BAAI/bge-reranker-v2-m3 qua sentence-transformers.

Luồng: top-20 từ Hybrid retriever → rerank → cắt còn top-5 (hoặc top-3).

Model:
  - BAAI/bge-reranker-v2-m3: cross-encoder multilingual, hỗ trợ tiếng Việt tốt.
  - Sử dụng CrossEncoder API của sentence-transformers (đơn giản hơn generative reranker).
  - Mỗi cặp (query, chunk) → 1 logit score → sort giảm dần.

GPU:
  - Nếu CUDA khả dụng: chạy fp16 tự động.
  - Nếu không: chạy CPU (chậm hơn ~10x nhưng vẫn chạy được với corpus nhỏ).
"""

from dataclasses import dataclass
from typing import List, Optional

from .base import RetrievalResult


@dataclass
class RerankerConfig:
    """Cấu hình cho CrossEncoderReranker.

    Attributes:
        model_name:  HuggingFace model ID.
        device:      "cuda" | "cpu" | "auto".
        max_length:  Độ dài tối đa token cho cross-encoder (mặc định 512).
        batch_size:  Số cặp (query, chunk) mỗi batch khi score.
    """
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "auto"
    max_length: int = 512
    batch_size: int = 32


class CrossEncoderReranker:
    """Reranker dùng cross-encoder để score từng cặp (query, chunk).

    Args:
        config: RerankerConfig.

    Ví dụ:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query="đái tháo đường thai kỳ",
                                   results=hybrid_results,
                                   top_k=5)
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        self._model = None

    # ------------------------------------------------------------------
    # Load / Unload
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load cross-encoder model vào bộ nhớ."""
        import torch
        from sentence_transformers import CrossEncoder

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[Reranker] Loading {self.config.model_name} → {device}...")
        self._model = CrossEncoder(
            self.config.model_name,
            max_length=self.config.max_length,
            device=device,
        )
        print(f"[Reranker] ✅ Model loaded.")

    def unload(self) -> None:
        """Giải phóng bộ nhớ."""
        import gc
        import torch
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[Reranker] Model unloaded.")

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Score và sắp xếp lại danh sách results theo cross-encoder score.

        Args:
            query:   Câu hỏi.
            results: Danh sách RetrievalResult từ retriever (thường 20–50 items).
            top_k:   Số kết quả trả về sau rerank.

        Returns:
            top_k RetrievalResult theo cross-encoder score giảm dần.
            Field `.score` được cập nhật thành logit score của cross-encoder.
        """
        if not results:
            return []

        if self._model is None:
            self.load()

        # Tạo danh sách cặp (query, chunk_text)
        pairs = [(query, r.text) for r in results]

        # Score tất cả cặp
        scores = self._model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        # Gán score mới và sort
        scored = [(r, float(s)) for r, s in zip(results, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for final_rank, (result, score) in enumerate(scored[:top_k]):
            reranked.append(RetrievalResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=score,
                metadata=result.metadata,
                rank=final_rank,
            ))
        return reranked
