from dataclasses import dataclass
from typing import Optional


@dataclass
class QdrantStoreConfig:
    """Cấu hình Qdrant cho pipeline đã chốt.

    Cấu hình benchmark chính thức:
        Chunker  : hierarchical
        Embedder : Qwen/Qwen3-Embedding-4B
        Distance : cosine
        Dim      : 2560 (output mặc định của Qwen3-Embedding-4B)

    Qdrant server/Docker:
        REST API : http://localhost:6333
        Dashboard: http://localhost:6333/dashboard
        gRPC     : localhost:6334

    Có thể dùng Qdrant local-file thay cho Docker bằng `path=...`.
    `memory=True` chỉ phù hợp test nhanh vì dữ liệu mất khi process kết thúc.
    """

    collection_name: str = "phu_san_chunks"
    host: str = "localhost"
    port: int = 6333

    # Qwen3-Embedding-4B default output dimension.
    # build_vector_store.py vẫn lấy dimension thật từ embeddings.shape[1]
    # và dùng giá trị đó để tạo collection, nên không phụ thuộc mù quáng vào số này.
    vector_size: int = 2560
    distance: str = "cosine"

    # Nếu path != None -> Qdrant embedded/local-file.
    path: Optional[str] = None

    # Nếu True -> Qdrant in-memory.
    memory: bool = False

    def __post_init__(self) -> None:
        if self.memory and self.path:
            raise ValueError("Chỉ chọn một trong hai: memory=True hoặc path=...")

        if self.vector_size <= 0:
            raise ValueError("vector_size phải > 0")

        self.distance = self.distance.lower().strip()
        if self.distance not in {"cosine", "dot", "euclidean"}:
            raise ValueError(
                "distance phải là một trong: cosine | dot | euclidean"
            )

        if self.port <= 0:
            raise ValueError("port phải > 0")

    @property
    def mode(self) -> str:
        if self.memory:
            return "memory"
        if self.path:
            return "local"
        return "server"

    @property
    def dashboard_url(self) -> Optional[str]:
        if self.mode != "server":
            return None
        return f"http://{self.host}:{self.port}/dashboard"
