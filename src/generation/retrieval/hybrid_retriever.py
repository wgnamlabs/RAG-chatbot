"""
HybridRetriever — Kết hợp Dense + BM25 bằng Reciprocal Rank Fusion (RRF).

Công thức RRF:
    RRF_score(doc) = Σ_retriever  weight_r / (k + rank_r(doc))

  - k = 60 (hằng số giúp giảm ảnh hưởng của rank rất cao)
  - weight_r: trọng số riêng cho từng retriever (mặc định = 1.0)
  - Nếu doc chỉ xuất hiện ở 1 retriever, chỉ cộng 1 term

Tài liệu tham khảo:
    Cormack, Clarke & Buettcher (2009). Reciprocal Rank Fusion Outperforms
    Condorcet and Individual Rank Learning Methods. SIGIR.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .base import BaseRetriever, RetrievalResult


@dataclass
class HybridRetrieverConfig:
    """Cấu hình cho HybridRetriever.

    Attributes:
        rrf_k:          Hằng số k trong công thức RRF (mặc định 60).
        dense_weight:   Trọng số cho dense retriever (mặc định 1.0).
        bm25_weight:    Trọng số cho BM25 retriever (mặc định 1.0).
        dense_top_k:    Số kết quả lấy từ dense trước khi merge (mặc định 50).
        bm25_top_k:     Số kết quả lấy từ BM25 trước khi merge (mặc định 50).
    """
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 3.0  # BM25 dominant (based on eval results)
    dense_top_k: int = 50
    bm25_top_k: int = 50


class HybridRetriever(BaseRetriever):
    """Hybrid retriever dùng manual Reciprocal Rank Fusion (RRF).

    Nhận kết quả từ dense_retriever và bm25_retriever, merge theo RRF,
    trả về top-k chunk có RRF score cao nhất.

    Args:
        dense_retriever: Một BaseRetriever trả về dense results.
        bm25_retriever:  Một BaseRetriever trả về BM25 results.
        config:          HybridRetrieverConfig.

    Ví dụ:
        hybrid = HybridRetriever(dense, bm25, config)
        results = hybrid.retrieve("đái tháo đường thai kỳ", top_k=10)
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        config: Optional[HybridRetrieverConfig] = None,
    ):
        self._dense = dense_retriever
        self._bm25  = bm25_retriever
        self.config = config or HybridRetrieverConfig()

    # ------------------------------------------------------------------
    # Core RRF
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_merge(
        ranked_lists: List[List[RetrievalResult]],
        weights: List[float],
        k: int = 60,
    ) -> List[RetrievalResult]:
        """Merge nhiều ranked list bằng weighted RRF.

        Args:
            ranked_lists: List of ranked result lists, mỗi list đã sorted theo score.
            weights:      Weight cho từng list (len == len(ranked_lists)).
            k:            Hằng số RRF.

        Returns:
            Danh sách RetrievalResult sorted theo RRF score giảm dần,
            với score = tổng weighted RRF, metadata từ lần xuất hiện đầu tiên.
        """
        # chunk_id → {rrf_score, text, metadata}
        rrf_scores: dict = {}
        chunk_meta: dict = {}

        for ranked_list, weight in zip(ranked_lists, weights):
            for rank, result in enumerate(ranked_list):
                cid = result.chunk_id
                contrib = weight / (k + rank + 1)  # rank là 0-indexed
                if cid not in rrf_scores:
                    rrf_scores[cid] = 0.0
                    chunk_meta[cid] = result
                rrf_scores[cid] += contrib

        # Sort theo RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        merged = []
        for final_rank, cid in enumerate(sorted_ids):
            orig = chunk_meta[cid]
            merged.append(RetrievalResult(
                chunk_id=cid,
                text=orig.text,
                score=rrf_scores[cid],
                metadata=orig.metadata,
                rank=final_rank,
            ))
        return merged

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Hybrid retrieval: Dense + BM25 → RRF merge → top-k.

        Args:
            query: Câu hỏi.
            top_k: Số kết quả cuối trả về.

        Returns:
            Danh sách top-k RetrievalResult theo RRF score.
        """
        cfg = self.config

        # Lấy kết quả từ từng retriever (nhiều hơn top_k để RRF có đủ input)
        dense_results = self._dense.retrieve(query, top_k=cfg.dense_top_k)
        bm25_results  = self._bm25.retrieve(query, top_k=cfg.bm25_top_k)

        merged = self._rrf_merge(
            ranked_lists=[dense_results, bm25_results],
            weights=[cfg.dense_weight, cfg.bm25_weight],
            k=cfg.rrf_k,
        )

        return merged[:top_k]
