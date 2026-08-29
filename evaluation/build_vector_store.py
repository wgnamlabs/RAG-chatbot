"""Build dense Qdrant index + BM25 index từ chunk cache đã benchmark.

Cấu hình mặc định được FREEZE từ kết quả DEV/TEST:
    Chunker  : hierarchical
    Embedder : Qwen/Qwen3-Embedding-4B
    Distance : cosine
    Top-K    : 15 candidate chunks

Benchmark TEST của cấu hình này:
    Evidence Recall@10   = 0.9786
    Evidence Complete@10 = 0.9786
    nDCG@10              = 0.8570
    MRR                  = 0.8434
    Evidence Recall@15   = 1.0000

Windows / Docker:
    docker run -d --name qdrant \
        -p 6333:6333 -p 6334:6334 \
        -v D:/rag-phu-san-chatbot/data/vector_db/qdrant:/qdrant/storage \
        qdrant/qdrant

    Dashboard:
        http://localhost:6333/dashboard

Build sạch sau khi đổi từ index cũ:
    python evaluation/build_vector_store.py --recreate

Kaggle:
    python evaluation/build_vector_store.py --qdrant-mode local --recreate

`auto` mode:
    - Kaggle/Colab -> local-file Qdrant
    - môi trường khác -> Qdrant server/Docker
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np


# Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from indexing.chunking.base import Chunk
from indexing.embedding import (
    MODELS_TO_COMPARE,
    EmbedderConfig,
    SentenceTransformerEmbedder,
)
from indexing.vector_store import (
    QdrantStoreConfig,
    QdrantVectorStore,
)


# ---------------------------------------------------------------------
# Frozen selection from DEV
# ---------------------------------------------------------------------

DEFAULT_CHUNKER = "hierarchical"
DEFAULT_EMB_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_EXPECTED_DIM = 2560
DEFAULT_COLLECTION = "phu_san_chunks"
DEFAULT_DISTANCE = "cosine"
DEFAULT_TOP_K = 15


def _is_notebook_cloud() -> bool:
    return (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "COLAB_RELEASE_TAG" in os.environ
    )


def _resolve_qdrant_mode(mode: str) -> str:
    if mode != "auto":
        return mode
    return "local" if _is_notebook_cloud() else "server"


def _selected_embedder_config(
    model_name: str,
    device: str,
) -> EmbedderConfig:
    """Dùng đúng config model đã dùng trong embedding benchmark."""
    for cfg in MODELS_TO_COMPARE:
        if cfg.model_name == model_name:
            return replace(cfg, device=device)

    # Cho phép experiment model khác, nhưng default/final vẫn là Qwen3-4B.
    print(
        f"[WARN] Model '{model_name}' không nằm trong MODELS_TO_COMPARE; "
        "dùng EmbedderConfig generic. Kết quả này không còn là cấu hình "
        "đã benchmark/freeze."
    )
    return EmbedderConfig(
        model_name=model_name,
        batch_size=16,
        max_seq_length=2048,
        device=device,
        trust_remote_code=False,
        auto_batch=True,
        min_batch_size=2,
    )


def _load_chunks(
    chunk_file: Path,
) -> Tuple[List[Chunk], list]:
    with chunk_file.open("r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError(
            f"{chunk_file} phải là JSON list không rỗng."
        )

    chunks: List[Chunk] = []
    chunk_ids = []

    for i, item in enumerate(raw_chunks):
        if not isinstance(item, dict):
            raise ValueError(f"Chunk #{i} không phải object.")
        if "text" not in item or "metadata" not in item:
            raise ValueError(
                f"Chunk #{i} thiếu 'text' hoặc 'metadata'."
            )

        metadata = item["metadata"] or {}
        chunk_id = metadata.get("chunk_id")
        if not chunk_id:
            raise ValueError(
                f"Chunk #{i} thiếu metadata.chunk_id. "
                "Hãy dùng output chunking final đã QA."
            )

        chunks.append(
            Chunk(
                text=item["text"],
                metadata=metadata,
            )
        )
        chunk_ids.append(str(chunk_id))

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Chunk cache có chunk_id trùng; dừng build để tránh Qdrant overwrite."
        )

    return chunks, raw_chunks


def _validate_embeddings(
    embeddings: np.ndarray,
    n_chunks: int,
    expected_dim: int | None,
) -> int:
    embeddings = np.asarray(embeddings)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embedding phải là matrix 2-D, nhận {embeddings.shape}"
        )

    if embeddings.shape[0] != n_chunks:
        raise ValueError(
            f"Embeddings rows={embeddings.shape[0]} "
            f"khác chunks={n_chunks}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings có NaN/Inf.")

    actual_dim = int(embeddings.shape[1])

    if expected_dim is not None and actual_dim != expected_dim:
        raise RuntimeError(
            f"Embedding dimension thực tế={actual_dim}, "
            f"nhưng expected={expected_dim}. "
            "Không tạo Qdrant collection vì config/model có thể đã thay đổi."
        )

    return actual_dim


def _build_vn_tokenizer():
    """Tokenizer thống nhất để serialize BM25 corpus."""
    try:
        from underthesea import word_tokenize

        print("[BM25] tokenizer=underthesea")

        def tokenize(text: str):
            return word_tokenize(
                str(text).lower(),
                format="text",
            ).split()

        return tokenize, "underthesea"

    except ImportError:
        pass

    try:
        from pyvi import ViTokenizer

        print("[BM25] tokenizer=pyvi (fallback)")

        def tokenize(text: str):
            return ViTokenizer.tokenize(
                str(text).lower()
            ).split()

        return tokenize, "pyvi"

    except ImportError:
        pass

    print("[BM25] tokenizer=whitespace (fallback)")

    def tokenize(text: str):
        return str(text).lower().split()

    return tokenize, "whitespace"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_stores(
    base_path: Path,
    chunker_name: str = DEFAULT_CHUNKER,
    emb_model: str = DEFAULT_EMB_MODEL,
    expected_dim: int | None = DEFAULT_EXPECTED_DIM,
    recreate: bool = False,
    device: str = "cuda",
    qdrant_mode: str = "auto",
    host: str = "localhost",
    port: int = 6333,
    local_path: Path | None = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> None:

    print("=" * 72)
    print("BUILD FINAL VECTOR STORE")
    print("=" * 72)
    print(f"Chunker        : {chunker_name}")
    print(f"Embedding      : {emb_model}")
    print(f"Distance       : {DEFAULT_DISTANCE}")
    print(f"Recommended K  : {DEFAULT_TOP_K}")
    print()

    if (
        chunker_name != DEFAULT_CHUNKER
        or emb_model != DEFAULT_EMB_MODEL
    ):
        print(
            "[WARN] Bạn đang build cấu hình khác cấu hình đã chốt trên DEV:\n"
            f"       selected={DEFAULT_CHUNKER} + {DEFAULT_EMB_MODEL}\n"
            f"       current ={chunker_name} + {emb_model}\n"
        )

    # ---------------------------------------------------------------
    # Step 1: exact final chunk cache
    # ---------------------------------------------------------------
    chunk_file = (
        base_path
        / "evaluation"
        / "results"
        / "chunks_cache"
        / f"{chunker_name}.json"
    )

    if not chunk_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {chunk_file}.\n"
            "Hãy tạo chunk cache trước khi build vector store."
        )

    print(f"[Step 1] Load chunks: {chunk_file}")
    chunks, _ = _load_chunks(chunk_file)
    print(f"         chunks={len(chunks)}")

    # ---------------------------------------------------------------
    # Step 2: document embeddings
    # IMPORTANT: is_query=False exactly as benchmark
    # ---------------------------------------------------------------
    emb_config = _selected_embedder_config(
        emb_model,
        device=device,
    )

    print("[Step 2] Embedding chunks")
    print(
        f"         batch_size={emb_config.batch_size}, "
        f"auto_batch={emb_config.auto_batch}, "
        f"max_seq_length={emb_config.max_seq_length}"
    )

    embedder = SentenceTransformerEmbedder(emb_config)
    try:
        embedder.load()
        embeddings = embedder.encode(
            [c.text for c in chunks],
            is_query=False,
        )
    finally:
        embedder.unload()

    actual_dim = _validate_embeddings(
        embeddings,
        n_chunks=len(chunks),
        expected_dim=expected_dim,
    )

    print(
        f"         shape={embeddings.shape}, dim={actual_dim}"
    )

    # ---------------------------------------------------------------
    # Step 3: Qdrant
    # ---------------------------------------------------------------
    resolved_mode = _resolve_qdrant_mode(qdrant_mode)

    qdrant_path = None
    memory = False

    if resolved_mode == "local":
        qdrant_path = local_path or (
            base_path
            / "data"
            / "vector_db"
            / "qdrant_local"
        )
        qdrant_path.mkdir(parents=True, exist_ok=True)
    elif resolved_mode == "memory":
        memory = True
    elif resolved_mode != "server":
        raise ValueError(
            "qdrant_mode phải là auto|server|local|memory"
        )

    qdrant_config = QdrantStoreConfig(
        collection_name=collection_name,
        host=host,
        port=port,
        vector_size=actual_dim,
        distance=DEFAULT_DISTANCE,
        path=str(qdrant_path) if qdrant_path else None,
        memory=memory,
    )

    print(
        f"[Step 3] Qdrant mode={qdrant_config.mode}, "
        f"collection={collection_name}"
    )

    store = QdrantVectorStore(qdrant_config)
    store.load()
    store.create_collection(recreate=recreate)
    store.add(
        chunks,
        embeddings,
        chunker_type=chunker_name,
    )
    store.persist()

    qdrant_info = store.collection_info()
    print(
        f"         points={qdrant_info['points_count']}, "
        f"dim={qdrant_info['vector_size']}"
    )

    # ---------------------------------------------------------------
    # Step 4: BM25 over the SAME hierarchical chunks
    # ---------------------------------------------------------------
    print("[Step 4] Build BM25 index")

    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "Thiếu rank-bm25. Cài bằng: pip install rank-bm25"
        ) from exc

    tokenize, tokenizer_name = _build_vn_tokenizer()
    tokenized_corpus = [
        tokenize(c.text)
        for c in chunks
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    vector_db_dir = base_path / "data" / "vector_db"
    vector_db_dir.mkdir(parents=True, exist_ok=True)

    bm25_path = vector_db_dir / "bm25_index.pkl"

    bm25_data = {
        "bm25": bm25,
        "chunks": [
            {
                "text": c.text,
                "metadata": c.metadata,
            }
            for c in chunks
        ],
        "chunk_ids": [
            c.metadata["chunk_id"]
            for c in chunks
        ],
        "chunker_type": chunker_name,
        "emb_model": emb_model,
        "tokenizer": tokenizer_name,
        "recommended_top_k": DEFAULT_TOP_K,
    }

    with bm25_path.open("wb") as f:
        pickle.dump(bm25_data, f)

    print(f"         saved={bm25_path}")

    # ---------------------------------------------------------------
    # Step 5: reproducibility manifest
    # ---------------------------------------------------------------
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_on_dev": (
            chunker_name == DEFAULT_CHUNKER
            and emb_model == DEFAULT_EMB_MODEL
        ),
        "chunker": chunker_name,
        "embedding_model": emb_model,
        "embedding_config": asdict(emb_config),
        "embedding_dimension": actual_dim,
        "distance": DEFAULT_DISTANCE,
        "recommended_dense_top_k": DEFAULT_TOP_K,
        "chunk_count": len(chunks),
        "chunk_cache": str(chunk_file.relative_to(base_path)),
        "chunk_cache_sha256": _sha256(chunk_file),
        "qdrant": qdrant_info,
        "bm25": {
            "path": str(bm25_path.relative_to(base_path)),
            "tokenizer": tokenizer_name,
        },
        "benchmark_test": {
            "evidence_recall_at_10": 0.9785714285714285,
            "evidence_complete_at_10": 0.9785714285714285,
            "ndcg_at_10": 0.8570159062555588,
            "mrr": 0.8433704390847248,
            "ood_auroc": 1.0,
            "evidence_recall_at_15": 1.0,
        },
    }

    manifest_path = vector_db_dir / "index_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[Step 5] Manifest: {manifest_path}")

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(
        f"Qdrant collection : {collection_name}\n"
        f"Qdrant mode       : {qdrant_config.mode}\n"
        f"Vector dimension  : {actual_dim}\n"
        f"Chunker           : {chunker_name}\n"
        f"Embedding         : {emb_model}\n"
        f"BM25              : {bm25_path}\n"
        f"Manifest          : {manifest_path}"
    )

    if qdrant_config.dashboard_url:
        print(
            f"Dashboard         : "
            f"{qdrant_config.dashboard_url}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build final Qdrant + BM25 index. "
            "Default = hierarchical + Qwen3-Embedding-4B."
        )
    )

    parser.add_argument(
        "--chunker",
        choices=["semantic", "hierarchical"],
        default=DEFAULT_CHUNKER,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMB_MODEL,
        help="Hugging Face embedding model.",
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=DEFAULT_EXPECTED_DIM,
        help=(
            "Chỉ dùng để validate output dimension. "
            "Qwen3-Embedding-4B mặc định = 2560."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Xóa collection cũ trước khi build. "
            "NÊN dùng khi chuyển từ index/model cũ."
        ),
    )

    parser.add_argument(
        "--qdrant-mode",
        choices=["auto", "server", "local", "memory"],
        default="auto",
        help=(
            "auto: Kaggle/Colab=local, môi trường khác=server."
        ),
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
    )
    parser.add_argument(
        "--host",
        default="localhost",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
    )
    parser.add_argument(
        "--local-path",
        type=Path,
        default=None,
        help="Path cho Qdrant local-file mode.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_stores(
        base_path=PROJECT_ROOT,
        chunker_name=args.chunker,
        emb_model=args.model,
        expected_dim=args.expected_dim,
        recreate=args.recreate,
        device=args.device,
        qdrant_mode=args.qdrant_mode,
        host=args.host,
        port=args.port,
        local_path=args.local_path,
        collection_name=args.collection,
    )
