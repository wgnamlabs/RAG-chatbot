"""
retrieval.py — Dense search, Sparse (BM25) search, và Hybrid RRF search.

Tokenizer BM25: underthesea (ưu tiên) → pyvi → str.split()
PHẢI dùng đúng tokenizer đã build BM25 (xem build_vector_store.py _build_vn_tokenizer).

RRF formula (rank bắt đầu từ 1):
    score(chunk) = sum(1 / (rank_i + rrf_k))   qua tất cả list chunk xuất hiện
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from generation.schemas import Chunk

logger = logging.getLogger(__name__)


# ── Vietnamese tokenizer (khớp với build_vector_store.py) ────────────────────

def _get_vn_tokenizer():
    """Trả về hàm tokenize tiếng Việt, ưu tiên underthesea → pyvi → split.

    Thứ tự PHẢI khớp với _build_vn_tokenizer() trong build_vector_store.py.
    """
    try:
        from underthesea import word_tokenize
        logger.debug("[retrieval] BM25 tokenizer: underthesea")
        return lambda text: word_tokenize(text, format="text").split()
    except ImportError:
        pass
    try:
        from pyvi import ViTokenizer
        logger.debug("[retrieval] BM25 tokenizer: pyvi (fallback)")
        return lambda text: ViTokenizer.tokenize(text).split()
    except ImportError:
        pass
    logger.warning("[retrieval] BM25 tokenizer: str.split() (fallback cuối — kết quả kém chính xác hơn)")
    return lambda text: text.lower().split()


_vn_tokenize = _get_vn_tokenizer()


# ── Helper: map Qdrant result dict → Chunk ───────────────────────────────────

def _qdrant_result_to_chunk(result: dict) -> Chunk:
    """Map 1 dict từ QdrantVectorStore.search() sang Chunk (generation schema).

    Payload Qdrant thực tế (xác nhận từ Dashboard):
        chunk_id, text, source_file, source, chunk_index, chunker_type,
        Header 2, is_table, too_long, breadcrumb

    section = breadcrumb nếu có, fallback Header 2.
    """
    metadata = result.get("metadata", {})
    breadcrumb = metadata.get("breadcrumb")
    # Header 2 có space trong tên key
    header2 = metadata.get("Header 2")
    section = breadcrumb or header2 or None

    return Chunk(
        chunk_id=result["chunk_id"],
        text=result["text"],
        source=metadata.get("source_file") or metadata.get("source", "unknown"),
        section=section,
        score=float(result["score"]),
    )


# ── Helper: map BM25 corpus entry → Chunk ────────────────────────────────────

def _bm25_entry_to_chunk(idx: int, entry: dict, score: float) -> Chunk:
    """Map 1 entry từ bm25_data["chunks"] sang Chunk.

    bm25_data["chunks"][i] = {"text": ..., "metadata": {...}}
    metadata từ semantic chunker có thể chứa: source, source_file, chunk_index,
    breadcrumb, Header 2, is_table, too_long, chunk_id, ...
    """
    meta = entry.get("metadata", {})

    # chunk_id: ưu tiên lấy từ metadata, fallback tự build
    chunk_id = (
        meta.get("chunk_id")
        or f"{meta.get('source', meta.get('source_file', 'unknown'))}::{idx}"
    )

    breadcrumb = meta.get("breadcrumb")
    header2 = meta.get("Header 2")
    section = breadcrumb or header2 or None

    source = meta.get("source_file") or meta.get("source", "unknown")

    return Chunk(
        chunk_id=chunk_id,
        text=entry["text"],
        source=source,
        section=section,
        score=score,
    )


# ── 1. dense_search ───────────────────────────────────────────────────────────

def dense_search(
    query: str,
    qdrant_store,          # QdrantVectorStore — tránh import vòng
    embedder,              # SentenceTransformerEmbedder
    top_k: int = 15,
) -> list[Chunk]:
    """Tìm kiếm vector (cosine similarity) trên Qdrant.

    Args:
        query:        Câu hỏi (gốc hoặc rewritten).
        qdrant_store: Instance QdrantVectorStore đã load().
        embedder:     Instance SentenceTransformerEmbedder đã load().
        top_k:        Số chunk trả về.

    Returns:
        List[Chunk] sort giảm dần theo cosine score.
        Trả về [] nếu Qdrant lỗi kết nối — không raise.
    """
    try:
        vec = embedder.encode([query], is_query=True)   # shape (1, 1024)
        results = qdrant_store.search(query_vector=vec[0], top_k=top_k)
    except Exception as exc:
        logger.error("[dense_search] Lỗi: %s. Trả về [].", exc)
        return []

    chunks = [_qdrant_result_to_chunk(r) for r in results]
    logger.debug("[dense_search] query=%r → %d chunks", query, len(chunks))
    return chunks


# ── 2. sparse_search (BM25) ──────────────────────────────────────────────────

def sparse_search(
    query: str,
    bm25_data: dict,
    top_k: int = 15,
) -> list[Chunk]:
    """Tìm kiếm sparse (BM25) trên corpus đã index.

    Args:
        query:     Câu hỏi (gốc hoặc rewritten).
        bm25_data: Dict load từ bm25_index.pkl:
                   {"bm25": BM25Okapi, "chunks": list[dict], ...}
        top_k:     Số chunk trả về.

    Returns:
        List[Chunk] sort giảm dần theo BM25 score.
        Trả về [] nếu tất cả score = 0 (không có từ khớp).
    """
    bm25 = bm25_data["bm25"]
    corpus_chunks: list[dict] = bm25_data["chunks"]

    tokenized = _vn_tokenize(query)
    scores: np.ndarray = bm25.get_scores(tokenized)   # shape (N,)

    if scores.max() == 0:
        logger.debug("[sparse_search] query=%r → không có từ khớp, trả về [].", query)
        return []

    # Lấy top_k index có điểm cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]

    chunks = [
        _bm25_entry_to_chunk(int(idx), corpus_chunks[idx], float(scores[idx]))
        for idx in top_indices
        if float(scores[idx]) > 0   # bỏ qua chunk điểm 0
    ]
    logger.debug("[sparse_search] query=%r → %d chunks", query, len(chunks))
    return chunks


# ── 3. hybrid_search (RRF) ───────────────────────────────────────────────────

def hybrid_search(
    original_query: str,
    rewritten_query: str,
    qdrant_store,
    bm25_data: dict,
    embedder,
    top_k: int = 15,
    rrf_k: int = 60,
) -> list[Chunk]:
    """Kết hợp 4 danh sách retrieval bằng Reciprocal Rank Fusion (RRF).

    4 danh sách:
        1. dense_search(original_query)
        2. dense_search(rewritten_query)
        3. sparse_search(original_query)
        4. sparse_search(rewritten_query)

    RRF formula (rank bắt đầu từ 1, không phải 0):
        rrf_score(chunk) = Σ  1 / (rank_i + rrf_k)

    Nếu 1 chunk_id xuất hiện ở nhiều list → CỘNG DỒN (không lấy max).
    Dedup theo chunk_id, sort giảm dần, trả về top_k.

    Args:
        original_query:  Câu hỏi gốc của người dùng.
        rewritten_query: Câu hỏi sau bước rewrite (có thể = original nếu fallback).
        qdrant_store:    Instance QdrantVectorStore đã load().
        bm25_data:       Dict từ bm25_index.pkl.
        embedder:        Instance SentenceTransformerEmbedder đã load().
        top_k:           Số chunk trả về sau RRF.
        rrf_k:           Hằng số RRF (default 60 theo paper gốc).

    Returns:
        List[Chunk] sort giảm dần theo rrf_score, len ≤ top_k.
        Chunk.score = rrf_score tổng hợp.
    """
    # ── Bước 1: gọi 4 lần retrieval ──────────────────────────────────────────
    lists: list[list[Chunk]] = [
        dense_search(original_query,  qdrant_store, embedder, top_k=top_k),
        dense_search(rewritten_query, qdrant_store, embedder, top_k=top_k),
        sparse_search(original_query,  bm25_data, top_k=top_k),
        sparse_search(rewritten_query, bm25_data, top_k=top_k),
    ]

    # ── Bước 2: tính RRF score và lưu chunk đại diện ─────────────────────────
    # rrf_scores[chunk_id] = tổng điểm RRF cộng dồn từ tất cả list
    rrf_scores: dict[str, float] = defaultdict(float)
    # chunk_store[chunk_id] = Chunk object đại diện (lấy từ list đầu tiên xuất hiện)
    chunk_store: dict[str, Chunk] = {}

    for rank_list in lists:
        for rank_one_indexed, chunk in enumerate(rank_list, start=1):  # rank bắt đầu từ 1
            rrf_scores[chunk.chunk_id] += 1.0 / (rank_one_indexed + rrf_k)
            if chunk.chunk_id not in chunk_store:
                chunk_store[chunk.chunk_id] = chunk

    # ── Bước 3: sort giảm dần theo rrf_score ─────────────────────────────────
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    # ── Bước 4: build kết quả, gán chunk.score = rrf_score ───────────────────
    results: list[Chunk] = []
    for cid in sorted_ids[:top_k]:
        chunk = chunk_store[cid]
        # Pydantic model là immutable by default — dùng model_copy để cập nhật score
        results.append(chunk.model_copy(update={"score": rrf_scores[cid]}))

    logger.debug(
        "[hybrid_search] original=%r, rewritten=%r → %d chunks sau RRF",
        original_query, rewritten_query, len(results),
    )
    return results


# ── Unit test RRF (chạy inline khi import với --test) ────────────────────────

def _unit_test_rrf() -> None:
    """Verify công thức RRF với con số cụ thể từ spec.

    Chunk "A" đứng rank 1 ở list 1 và rank 3 ở list 2:
        score = 1/(1+60) + 1/(3+60) = 0.016393... + 0.015873... = 0.032267...
    Assert sai số < 1e-4.
    """
    expected = 1.0 / (1 + 60) + 1.0 / (3 + 60)  # 0.032267...

    # Tạo mock chunks để test công thức thuần túy
    rrf_scores: dict[str, float] = defaultdict(float)

    # List 1: A ở rank 1, B ở rank 2
    mock_list_1 = ["A", "B"]
    for rank_one, cid in enumerate(mock_list_1, start=1):
        rrf_scores[cid] += 1.0 / (rank_one + 60)

    # List 2: B ở rank 1, C ở rank 2, A ở rank 3
    mock_list_2 = ["B", "C", "A"]
    for rank_one, cid in enumerate(mock_list_2, start=1):
        rrf_scores[cid] += 1.0 / (rank_one + 60)

    score_A = rrf_scores["A"]
    assert abs(score_A - expected) < 1e-4, (
        f"RRF score sai: got {score_A:.6f}, expected {expected:.6f}"
    )
    print(f"  [RRF unit test] score_A = {score_A:.6f} ≈ {expected:.6f} ✓")
    print(f"  [RRF unit test] 1/(1+60) = {1/(1+60):.6f}")
    print(f"  [RRF unit test] 1/(3+60) = {1/(3+60):.6f}")
