"""
BM25Retriever — Tìm kiếm từ khóa bằng BM25Okapi với tokenize tiếng Việt.

Tokenizer ưu tiên (theo thứ tự):
  1. underthesea.word_tokenize — chất lượng tốt nhất cho tiếng Việt y khoa
  2. pyvi.ViTokenizer           — nhẹ hơn, fallback nếu underthesea lỗi
  3. str.split()                — fallback cuối cùng (không tách từ ghép)

BM25 index có thể được tạo mới từ chunks, hoặc load từ file pickle đã build sẵn
bởi evaluation/build_vector_store.py.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Callable

from .base import BaseRetriever, RetrievalResult


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _build_vn_tokenizer() -> Callable[[str], List[str]]:
    """Tạo hàm tokenize tiếng Việt tốt nhất khả dụng."""
    try:
        from underthesea import word_tokenize
        return lambda text: word_tokenize(text.lower(), format="text").split()
    except ImportError:
        pass
    try:
        from pyvi import ViTokenizer
        return lambda text: ViTokenizer.tokenize(text.lower()).split()
    except ImportError:
        pass
    return lambda text: text.lower().split()


# ── BM25Retriever ─────────────────────────────────────────────────────────────

class BM25Retriever(BaseRetriever):
    """Retriever dùng BM25Okapi từ thư viện rank_bm25.

    Args:
        chunks:          Danh sách dict {"text": str, "metadata": dict}.
                         Nếu None, phải gọi load_from_pickle() sau khi khởi tạo.
        tokenizer_fn:    Hàm tokenize. Mặc định dùng _build_vn_tokenizer().
        k1, b:           Tham số BM25Okapi (mặc định: k1=1.5, b=0.75).

    Ví dụ:
        retriever = BM25Retriever(chunks=my_chunks)
        results = retriever.retrieve("đái tháo đường thai kỳ", top_k=10)
    """

    def __init__(
        self,
        chunks: Optional[List[dict]] = None,
        tokenizer_fn: Optional[Callable] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self._tokenize: Callable = tokenizer_fn or _build_vn_tokenizer()
        self._chunks: List[dict] = []
        self._bm25 = None
        self.k1 = k1
        self.b = b

        if chunks is not None:
            self._build_index(chunks)

    # ------------------------------------------------------------------
    # Build / Load index
    # ------------------------------------------------------------------

    def _build_index(self, chunks: List[dict]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        print(f"[BM25] Building index từ {len(chunks)} chunks...")
        tokenized = [self._tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        print(f"[BM25] ✅ Index ready.")

    def load_from_pickle(self, pickle_path: Path) -> None:
        """Load BM25 index đã build sẵn từ file pickle."""
        print(f"[BM25] Loading index từ {pickle_path}...")
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        self._bm25  = data["bm25"]
        self._chunks = data["chunks"]
        print(f"[BM25] ✅ Loaded {len(self._chunks)} chunks từ pickle.")

    def save_to_pickle(self, pickle_path: Path) -> None:
        """Lưu BM25 index và chunks ra file pickle."""
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bm25":   self._bm25,
            "chunks": self._chunks,
        }
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[BM25] Saved index → {pickle_path}")

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Trả về top-k chunk có BM25 score cao nhất cho query.

        Args:
            query: Câu hỏi (sẽ được tokenize bằng cùng tokenizer với index).
            top_k: Số kết quả tối đa.

        Returns:
            Danh sách RetrievalResult, sắp xếp theo score giảm dần.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index chưa được build. Gọi _build_index() hoặc load_from_pickle() trước.")

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Lấy top-k index theo score giảm dần
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            chunk = self._chunks[idx]
            meta = chunk["metadata"].copy()
            chunk_id = meta.get("chunk_id") or f"{meta.get('source', 'unknown')}::{meta.get('chunk_index', idx)}"
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=chunk["text"],
                score=float(scores[idx]),
                metadata=meta,
                rank=rank,
            ))

        return results
