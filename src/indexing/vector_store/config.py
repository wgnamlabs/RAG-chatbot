from dataclasses import dataclass


@dataclass
class QdrantStoreConfig:
    """Cấu hình cho QdrantVectorStore kết nối tới Qdrant server (Docker).

    Chạy Qdrant bằng Docker:
        docker run -p 6333:6333 -p 6334:6334 \\
            -v D:/rag-phu-san-chatbot/data/vector_db/qdrant:/qdrant/storage \\
            qdrant/qdrant

    Xem dashboard tại: http://localhost:6334/dashboard

    Attributes:
        collection_name: Tên collection trong Qdrant.
        host:            Host của Qdrant server (mặc định localhost).
        port:            Port REST API (mặc định 6333).
        vector_size:     Số chiều embedding.
                         AITeamVN/Vietnamese_Embedding (based on bge-m3) → 1024.
        distance:        Hàm đo khoảng cách: "cosine" | "dot" | "euclidean".
    """

    collection_name: str = "phu_san_chunks"
    host: str = "localhost"
    port: int = 6333
    vector_size: int = 1024       # AITeamVN/Vietnamese_Embedding dim
    distance: str = "cosine"
    
    # Kaggle/Colab support:
    path: str = None              # Ví dụ: "data/vector_db/qdrant_local"
    memory: bool = False          # True nếu muốn chạy in-memory hoàn toàn
