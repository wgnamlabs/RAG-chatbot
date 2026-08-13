"""
postprocess.py — Hậu xử lý sau rerank: dedup_redundant + sandwich_order.

Thứ tự trong pipeline:
    rerank(top_k=10) → dedup_redundant(final_top_k=5) → sandwich_order() → build_prompt()

dedup_redundant:
  - Encode text từng chunk bằng embedder đã có (không load model mới).
  - Tính cosine similarity pairwise.
  - Nếu 2 chunk sim > threshold → giữ chunk đứng trước (score cao hơn), loại chunk sau.
  - Cắt còn final_top_k sau khi dedup.

sandwich_order (Lost-in-the-Middle mitigation):
  - Chunk tốt nhất (rank 1) đặt ở đầu.
  - Chunk tốt thứ 2 (rank 2) đặt ở cuối.
  - Các chunk còn lại giữ nguyên thứ tự ở giữa.
  - Ví dụ: [A,B,C,D,E] → [A,C,D,E,B]
"""

from __future__ import annotations

import logging

import numpy as np

from generation.schemas import Chunk

logger = logging.getLogger(__name__)


# ── dedup_redundant ──────────────────────────────────────────────────────────

def dedup_redundant(
    chunks: list[Chunk],
    embedder,                   # SentenceTransformerEmbedder (đã load)
    sim_threshold: float = 0.9,
    final_top_k: int = 5,
) -> list[Chunk]:
    """Loại bỏ chunk quá giống nhau rồi cắt còn final_top_k.

    Args:
        chunks:        Danh sách Chunk đã sort giảm dần theo rerank score.
        embedder:      SentenceTransformerEmbedder đã load() — KHÔNG load lại.
        sim_threshold: Ngưỡng cosine similarity để coi là trùng (mặc định 0.9).
        final_top_k:   Số chunk giữ lại sau dedup và cắt (mặc định 5).

    Returns:
        List[Chunk] sau khi dedup, cắt còn final_top_k, giữ nguyên thứ tự sort.
    """
    if len(chunks) == 0:
        return []

    if len(chunks) == 1:
        return chunks[:final_top_k]

    # ── Encode tất cả chunk text ──────────────────────────────────────────────
    texts = [c.text for c in chunks]
    vecs: np.ndarray = embedder.encode(texts, is_query=False)  # (N, dim)

    # Normalize để tính cosine bằng dot product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)   # tránh chia cho 0
    vecs_norm = vecs / norms                      # (N, dim) normalized

    # ── Greedy dedup: duyệt từ đầu, loại chunk sau nếu sim > threshold ───────
    keep_mask = [True] * len(chunks)

    for i in range(len(chunks)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(chunks)):
            if not keep_mask[j]:
                continue
            sim = float(np.dot(vecs_norm[i], vecs_norm[j]))
            if sim > sim_threshold:
                # Chunk i (score cao hơn, đứng trước) được giữ — loại chunk j
                keep_mask[j] = False
                logger.debug(
                    "[dedup] Loại chunk[%d] (sim=%.4f > %.2f với chunk[%d])",
                    j, sim, sim_threshold, i,
                )

    kept = [c for c, keep in zip(chunks, keep_mask) if keep]

    logger.info(
        "[dedup] %d chunks → dedup (threshold=%.2f) → %d chunks → cắt top_%d",
        len(chunks), sim_threshold, len(kept), final_top_k,
    )

    return kept[:final_top_k]


# ── sandwich_order ───────────────────────────────────────────────────────────

def sandwich_order(chunks: list[Chunk]) -> list[Chunk]:
    """Sắp xếp chunk theo pattern "sandwich" để giảm Lost-in-the-Middle effect.

    Chunk quan trọng nhất (rank 1) được đặt ở đầu prompt.
    Chunk quan trọng thứ 2 (rank 2) được đặt ở cuối prompt.
    Các chunk còn lại nằm ở giữa, giữ nguyên thứ tự.

    Lý thuyết: LLM có xu hướng chú ý nhiều hơn vào đầu và cuối context
    (Lost in the Middle — Liu et al. 2023). Sandwich đặt 2 chunk quan trọng
    nhất ở 2 vị trí "mạnh" nhất.

    Ví dụ:
        Input:  [A, B, C, D, E]   (A = tốt nhất, B = tốt thứ 2)
        Output: [A, C, D, E, B]

    Args:
        chunks: Danh sách Chunk đã sort giảm dần (chunks[0] = tốt nhất).

    Returns:
        Danh sách Chunk được sắp xếp lại theo pattern sandwich.
        Nếu len(chunks) <= 2: trả về nguyên vẹn.
    """
    if len(chunks) <= 2:
        return chunks

    best   = chunks[0]      # rank 1 → đầu prompt
    second = chunks[1]      # rank 2 → cuối prompt
    middle = chunks[2:]     # còn lại → giữa, giữ thứ tự

    result = [best] + middle + [second]

    logger.debug(
        "[sandwich] %d chunks: [%s, ...(x%d)..., %s]",
        len(chunks),
        best.chunk_id[:20],
        len(middle),
        second.chunk_id[:20],
    )
    return result
