"""
QdrantVectorStore — Kết nối tới Qdrant server chạy qua Docker.

Khởi động Qdrant (chạy 1 lần, giữ terminal mở):
    docker run -p 6333:6333 -p 6334:6334 \\
        -v D:/rag-phu-san-chatbot/data/vector_db/qdrant:/qdrant/storage \\
        qdrant/qdrant

Sau đó:
  - REST API:   http://localhost:6333
  - Dashboard:  http://localhost:6334/dashboard  ← xem collection, vector, filter trực tiếp

Metadata payload mỗi point:
    - chunk_id     : str  — "{source_file}::{chunk_index}"
    - text         : str  — nội dung chunk
    - source_file  : str  — tên file .md nguồn
    - chunk_index  : int  — thứ tự chunk trong file
    - chunker_type : str  — "semantic" | "hierarchical"
    - (các trường metadata khác từ chunker)
"""

from typing import List, Optional

import numpy as np

from .base import BaseVectorStore
from .config import QdrantStoreConfig
from ..chunking.base import Chunk


class QdrantVectorStore(BaseVectorStore):
    """Vector store kết nối tới Qdrant server (Docker).

    Dữ liệu được persist tự động qua Docker volume mount.
    Xem và filter collection trực tiếp tại http://localhost:6334/dashboard.
    """

    def __init__(self, config: QdrantStoreConfig = None):
        self.config = config or QdrantStoreConfig()
        self._client = None

    # ------------------------------------------------------------------
    # Kết nối
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Kết nối tới Qdrant server tại host:port."""
        from qdrant_client import QdrantClient

        self._client = QdrantClient(
            host=self.config.host,
            port=self.config.port,
        )
        # Kiểm tra kết nối
        try:
            self._client.get_collections()
            print(f"[QdrantStore] ✅ Kết nối Qdrant tại "
                  f"http://{self.config.host}:{self.config.port}")
            print(f"[QdrantStore] 🌐 Dashboard: "
                  f"http://{self.config.host}:{self.config.port + 1}/dashboard")
        except Exception as e:
            raise ConnectionError(
                f"Không kết nối được Qdrant tại {self.config.host}:{self.config.port}.\n"
                f"Hãy khởi động Docker container trước:\n\n"
                f"  docker run -p 6333:6333 -p 6334:6334 \\\n"
                f"    -v D:/rag-phu-san-chatbot/data/vector_db/qdrant:/qdrant/storage \\\n"
                f"    qdrant/qdrant\n\n"
                f"Chi tiết lỗi: {e}"
            )

    def _ensure_loaded(self) -> None:
        if self._client is None:
            self.load()

    # ------------------------------------------------------------------
    # Tạo collection
    # ------------------------------------------------------------------

    def create_collection(self, recreate: bool = False) -> None:
        """Tạo collection. Nếu recreate=True, xoá và tạo lại."""
        from qdrant_client.models import VectorParams, Distance

        self._ensure_loaded()

        distance_map = {
            "cosine":    Distance.COSINE,
            "dot":       Distance.DOT,
            "euclidean": Distance.EUCLID,
        }
        distance = distance_map.get(self.config.distance, Distance.COSINE)

        existing = [c.name for c in self._client.get_collections().collections]

        if self.config.collection_name in existing:
            if recreate:
                print(f"[QdrantStore] Xoá collection cũ '{self.config.collection_name}'...")
                self._client.delete_collection(self.config.collection_name)
            else:
                print(f"[QdrantStore] Collection '{self.config.collection_name}' đã tồn tại, bỏ qua.")
                return

        self._client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(
                size=self.config.vector_size,
                distance=distance,
            ),
        )
        print(f"[QdrantStore] Tạo collection '{self.config.collection_name}' "
              f"(dim={self.config.vector_size}, distance={self.config.distance})")

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        chunker_type: str = "unknown",
        batch_size: int = 256,
    ) -> None:
        """Index danh sách chunks vào Qdrant.

        Args:
            chunks:       Danh sách Chunk objects.
            embeddings:   Numpy array (N, dim).
            chunker_type: "semantic" | "hierarchical".
            batch_size:   Số point mỗi lần upsert.
        """
        from qdrant_client.models import PointStruct

        self._ensure_loaded()

        assert len(chunks) == embeddings.shape[0], (
            f"Số chunks ({len(chunks)}) ≠ số embeddings ({embeddings.shape[0]})"
        )

        points = []
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            source_file = chunk.metadata.get("source", "unknown")
            chunk_index = chunk.metadata.get("chunk_index", idx)
            chunk_id_str = f"{source_file}::{chunk_index}"

            payload = {
                "chunk_id":     chunk_id_str,
                "text":         chunk.text,
                "source_file":  source_file,
                "chunk_index":  chunk_index,
                "chunker_type": chunker_type,
            }
            for k, v in chunk.metadata.items():
                if k not in payload and not callable(v):
                    payload[k] = v

            points.append(PointStruct(
                id=idx,
                vector=emb.tolist(),
                payload=payload,
            ))

        total = len(points)
        for start in range(0, total, batch_size):
            batch = points[start:start + batch_size]
            self._client.upsert(
                collection_name=self.config.collection_name,
                points=batch,
            )
            print(f"[QdrantStore] Upsert {min(start + batch_size, total)}/{total}...")

        print(f"[QdrantStore] ✅ Indexed {total} chunks → '{self.config.collection_name}'")
        print(f"[QdrantStore] 🌐 Xem tại: http://{self.config.host}:{self.config.port + 1}/dashboard")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[dict]:
        """Tìm kiếm top-k chunk gần nhất với query vector.

        Dùng query_points() thay vì search() (deprecated từ qdrant-client >= 1.7.0).
        """
        self._ensure_loaded()

        response = self._client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )

        output = []
        for hit in response.points:
            payload  = hit.payload or {}
            metadata = {k: v for k, v in payload.items() if k != "text"}
            output.append({
                "chunk_id": payload.get("chunk_id", str(hit.id)),
                "text":     payload.get("text", ""),
                "score":    hit.score,
                "metadata": metadata,
            })
        return output

    # ------------------------------------------------------------------
    # Persist / Info
    # ------------------------------------------------------------------

    def persist(self) -> None:
        """No-op — Qdrant server tự persist qua Docker volume."""
        print("[QdrantStore] Docker mode: dữ liệu persist qua volume mount.")

    def collection_info(self) -> dict:
        """Trả về thông tin collection."""
        self._ensure_loaded()
        info = self._client.get_collection(self.config.collection_name)
        return {
            "collection_name": self.config.collection_name,
            "points_count":    info.points_count,
            "vector_size":     self.config.vector_size,
            "distance":        self.config.distance,
            "dashboard":       f"http://{self.config.host}:{self.config.port + 1}/dashboard",
        }
