"""
rerank.py — CrossEncoder reranker dùng BAAI/bge-reranker-v2-m3.

Dùng sentence_transformers.CrossEncoder — KHÔNG dùng FlagEmbedding riêng.
bge-reranker-v2-m3 hỗ trợ đầy đủ qua CrossEncoder API.

Nguyên tắc:
  - top_k mặc định = 10 (không phải 5) để dedup_redundant có pool đủ lớn.
  - Pipeline sẽ cắt còn 5 SAU khi dedup_redundant chạy xong.
  - Nếu chunks rỗng → trả về [] ngay, không load model.
  - CrossEncoder được load 1 lần duy nhất (caller chịu trách nhiệm cache instance).
"""

from __future__ import annotations

import logging

from generation.schemas import Chunk

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    chunks: list[Chunk],
    top_k: int = 10,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "cpu",
) -> list[Chunk]:
    """Rerank danh sách chunk bằng CrossEncoder.

    Args:
        query:      Câu hỏi (nên dùng rewritten_query để khớp với retrieval).
        chunks:     Danh sách Chunk từ hybrid_search (hoặc dense_search).
        top_k:      Số chunk trả về sau rerank (mặc định 10 để dedup có pool đủ).
        model_name: HuggingFace model id của CrossEncoder.
        device:     "cuda" hoặc "cpu".

    Returns:
        List[Chunk] sort giảm dần theo rerank score, len ≤ top_k.
        chunk.score được GHI ĐÈ = điểm CrossEncoder (không cộng dồn với RRF score cũ).
        Trả về [] nếu input rỗng.
    """
    if not chunks:
        logger.debug("[rerank] Nhận chunks rỗng, trả về [] ngay.")
        return []

    # Load CrossEncoder (caller nên cache instance ngoài vòng lặp)
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers chưa cài. Chạy: pip install sentence-transformers"
        ) from exc

    logger.info("[rerank] Load CrossEncoder %s → %s ...", model_name, device)
    cross_encoder = CrossEncoder(model_name, device=device)

    # Tạo danh sách cặp (query, chunk.text)
    pairs = [(query, c.text) for c in chunks]

    # Predict scores — trả về list[float]
    scores: list[float] = cross_encoder.predict(pairs).tolist()

    # Gán score mới vào từng chunk (ghi đè score RRF cũ)
    reranked = [
        chunk.model_copy(update={"score": float(score)})
        for chunk, score in zip(chunks, scores)
    ]

    # Sort giảm dần theo rerank score
    reranked.sort(key=lambda c: c.score, reverse=True)

    result = reranked[:top_k]
    logger.info(
        "[rerank] %d chunks → rerank → top %d | top score=%.4f",
        len(chunks), len(result), result[0].score if result else 0.0,
    )
    return result
