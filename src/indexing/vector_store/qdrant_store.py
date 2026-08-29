"""Qdrant vector store cho pipeline RAG phụ sản.

Cấu hình production/retrieval đã chốt:
- Hierarchical chunks
- Qwen/Qwen3-Embedding-4B
- Cosine similarity
- Dense candidate pool mặc định: top_k=15

Qdrant Docker mặc định:
    REST API : http://localhost:6333
    Dashboard: http://localhost:6333/dashboard
    gRPC     : localhost:6334

Payload của mỗi point giữ nguyên metadata từ chunker, bao gồm:
- chunk_id
- source / source_file
- chunk_index
- breadcrumb / Header ...
- is_table
- token_count / content_token_count
- split metadata
- chunker_type
"""

from __future__ import annotations

import uuid
from typing import List

import numpy as np

from .base import BaseVectorStore
from .config import QdrantStoreConfig
from ..chunking.base import Chunk


_POINT_NAMESPACE = uuid.UUID("f629e049-f76b-47fe-b3a6-d6eb2721a36d")


class QdrantVectorStore(BaseVectorStore):
    """Qdrant backend dùng được ở server, local-file hoặc memory mode."""

    def __init__(self, config: QdrantStoreConfig | None = None):
        self.config = config or QdrantStoreConfig()
        self._client = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Khởi tạo/kết nối Qdrant theo config."""
        from qdrant_client import QdrantClient

        if self._client is not None:
            return

        if self.config.memory:
            self._client = QdrantClient(":memory:")
            print("[QdrantStore] ✅ Qdrant in-memory")
        elif self.config.path:
            self._client = QdrantClient(path=self.config.path)
            print(
                f"[QdrantStore] ✅ Qdrant local-file: {self.config.path}"
            )
        else:
            self._client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
            )

        try:
            self._client.get_collections()
        except Exception as exc:
            if self.config.mode == "server":
                raise ConnectionError(
                    "Không kết nối được Qdrant server tại "
                    f"{self.config.host}:{self.config.port}.\n"
                    "Hãy kiểm tra Docker/Qdrant trước khi build index.\n"
                    "REST + Dashboard dùng port 6333; gRPC dùng port 6334.\n"
                    f"Chi tiết: {exc}"
                ) from exc
            raise

        if self.config.mode == "server":
            print(
                f"[QdrantStore] ✅ Connected: "
                f"http://{self.config.host}:{self.config.port}"
            )
            print(
                f"[QdrantStore] 🌐 Dashboard: "
                f"{self.config.dashboard_url}"
            )

    def _ensure_loaded(self) -> None:
        if self._client is None:
            self.load()

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    @staticmethod
    def _distance_name(value) -> str:
        """Chuẩn hóa Distance enum/string từ nhiều qdrant-client versions."""
        if value is None:
            return ""
        raw = getattr(value, "value", value)
        return str(raw).lower()

    def _existing_vector_config(self):
        info = self._client.get_collection(self.config.collection_name)
        vectors = info.config.params.vectors

        # Pipeline hiện tại chỉ dùng single-vector collection.
        if isinstance(vectors, dict):
            raise RuntimeError(
                f"Collection '{self.config.collection_name}' là named/multi-vector "
                "collection, không tương thích với pipeline single-vector hiện tại. "
                "Hãy build lại bằng --recreate."
            )

        return {
            "size": int(vectors.size),
            "distance": self._distance_name(vectors.distance),
        }

    def create_collection(self, recreate: bool = False) -> None:
        """Tạo collection và chặn reuse collection sai dimension/distance.

        Điều này đặc biệt quan trọng khi chuyển từ index cũ 1024 chiều
        sang Qwen3-Embedding-4B 2560 chiều.
        """
        from qdrant_client.models import Distance, VectorParams

        self._ensure_loaded()

        distance_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID,
        }
        expected_distance = self.config.distance
        expected_size = self.config.vector_size

        existing = {
            c.name for c in self._client.get_collections().collections
        }

        if self.config.collection_name in existing:
            if recreate:
                print(
                    f"[QdrantStore] Xóa collection cũ "
                    f"'{self.config.collection_name}'..."
                )
                self._client.delete_collection(
                    self.config.collection_name
                )
            else:
                current = self._existing_vector_config()
                if (
                    current["size"] != expected_size
                    or current["distance"] != expected_distance
                ):
                    raise RuntimeError(
                        "Collection Qdrant hiện tại không tương thích:\n"
                        f"  existing: dim={current['size']}, "
                        f"distance={current['distance']}\n"
                        f"  expected: dim={expected_size}, "
                        f"distance={expected_distance}\n"
                        "Bạn vừa đổi embedding/config hoặc đang dùng index cũ. "
                        "Hãy chạy build_vector_store.py với --recreate."
                    )

                print(
                    f"[QdrantStore] Collection "
                    f"'{self.config.collection_name}' đã tồn tại và "
                    "đúng cấu hình; sẽ upsert/update points."
                )
                return

        self._client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(
                size=expected_size,
                distance=distance_map[expected_distance],
            ),
        )

        print(
            f"[QdrantStore] ✅ Created collection "
            f"'{self.config.collection_name}' "
            f"(dim={expected_size}, distance={expected_distance})"
        )

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_id(chunk: Chunk, fallback_index: int) -> str:
        metadata = chunk.metadata or {}

        # Ưu tiên chunk_id đã được chunker tạo và QA là globally unique.
        existing = metadata.get("chunk_id")
        if existing:
            return str(existing)

        source_file = metadata.get(
            "source",
            metadata.get("source_file", "unknown"),
        )
        chunk_index = metadata.get("chunk_index", fallback_index)
        return f"{source_file}::{chunk_index}"

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Qdrant point ID ổn định, không phụ thuộc thứ tự list."""
        return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))

    def add(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        chunker_type: str = "hierarchical",
        batch_size: int = 128,
    ) -> None:
        """Index chunks vào Qdrant.

        Dùng deterministic UUID từ `chunk_id`, nên rebuild/upsert cùng corpus
        không tạo point ID mới theo thứ tự ngẫu nhiên.
        """
        from qdrant_client.models import PointStruct

        self._ensure_loaded()

        embeddings = np.asarray(embeddings)

        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings phải là matrix 2-D, nhận shape={embeddings.shape}"
            )

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Số chunks ({len(chunks)}) != "
                f"số embeddings ({embeddings.shape[0]})"
            )

        if embeddings.shape[1] != self.config.vector_size:
            raise ValueError(
                f"Embedding dimension={embeddings.shape[1]} nhưng "
                f"Qdrant config vector_size={self.config.vector_size}."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings chứa NaN hoặc Inf.")

        chunk_ids = [
            self._chunk_id(chunk, i)
            for i, chunk in enumerate(chunks)
        ]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "Phát hiện chunk_id trùng. Không index để tránh ghi đè point."
            )

        total = len(chunks)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            points = []

            for idx in range(start, end):
                chunk = chunks[idx]
                metadata = dict(chunk.metadata or {})
                chunk_id = chunk_ids[idx]

                source_file = metadata.get(
                    "source",
                    metadata.get("source_file", "unknown"),
                )
                chunk_index = metadata.get("chunk_index", idx)

                payload = dict(metadata)
                payload.update({
                    "chunk_id": chunk_id,
                    "text": chunk.text,
                    "source_file": source_file,
                    "chunk_index": chunk_index,
                    "chunker_type": chunker_type,
                })

                points.append(
                    PointStruct(
                        id=self._point_id(chunk_id),
                        vector=embeddings[idx].tolist(),
                        payload=payload,
                    )
                )

            self._client.upsert(
                collection_name=self.config.collection_name,
                points=points,
                wait=True,
            )

            print(
                f"[QdrantStore] Upsert {end}/{total}..."
            )

        print(
            f"[QdrantStore] ✅ Indexed {total} chunks -> "
            f"'{self.config.collection_name}'"
        )

        if self.config.dashboard_url:
            print(
                f"[QdrantStore] 🌐 Dashboard: "
                f"{self.config.dashboard_url}"
            )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 15,
    ) -> List[dict]:
        """Dense retrieval bằng cosine, mặc định candidate pool Top-15."""
        self._ensure_loaded()

        if top_k <= 0:
            raise ValueError("top_k phải > 0")

        vector = np.asarray(query_vector).reshape(-1)

        if vector.shape[0] != self.config.vector_size:
            raise ValueError(
                f"Query vector dim={vector.shape[0]} nhưng collection "
                f"dim={self.config.vector_size}."
            )

        if not np.isfinite(vector).all():
            raise ValueError("Query vector chứa NaN hoặc Inf.")

        response = self._client.query_points(
            collection_name=self.config.collection_name,
            query=vector.tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        output = []
        for hit in response.points:
            payload = hit.payload or {}
            metadata = {
                k: v
                for k, v in payload.items()
                if k != "text"
            }

            output.append({
                "chunk_id": payload.get(
                    "chunk_id",
                    str(hit.id),
                ),
                "text": payload.get("text", ""),
                "score": float(hit.score),
                "metadata": metadata,
            })

        return output

    # ------------------------------------------------------------------
    # Persist / Info
    # ------------------------------------------------------------------

    def persist(self) -> None:
        """Qdrant server/local-file tự persist; memory mode không persist."""
        if self.config.mode == "memory":
            print(
                "[QdrantStore] memory mode: dữ liệu sẽ mất khi process kết thúc."
            )
        elif self.config.mode == "local":
            print(
                f"[QdrantStore] local-file mode: persisted tại "
                f"{self.config.path}"
            )
        else:
            print(
                "[QdrantStore] server mode: Qdrant tự persist theo storage config."
            )

    def collection_info(self) -> dict:
        self._ensure_loaded()

        info = self._client.get_collection(
            self.config.collection_name
        )
        current = self._existing_vector_config()

        return {
            "collection_name": self.config.collection_name,
            "points_count": info.points_count,
            "vector_size": current["size"],
            "distance": current["distance"],
            "mode": self.config.mode,
            "dashboard": self.config.dashboard_url,
        }
