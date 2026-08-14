import sys
import pickle
from pathlib import Path

# Reconfigure stdout for Windows console
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from indexing.embedding import SentenceTransformerEmbedder
from indexing.embedding.config import EmbedderConfig
from indexing.vector_store import QdrantVectorStore, QdrantStoreConfig
from generation.pipeline.main import run_pipeline

def main():
    print("⏳ Đang tải mô hình và kết nối database (chỉ tốn vài giây)...")
    # Load infra
    embedder = SentenceTransformerEmbedder(EmbedderConfig(
        model_name="AITeamVN/Vietnamese_Embedding_v2", device="cuda"
    ))
    embedder.load()

    qdrant = QdrantVectorStore(config=QdrantStoreConfig(collection_name="phu_san_chunks"))
    qdrant.load()

    with open("data/vector_db/bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)

    print("✅ Hệ thống đã sẵn sàng!")
    print("-" * 50)

    while True:
        try:
            query = input("\n🧑 Bạn hỏi: ")
            if not query.strip():
                continue
            if query.lower() in ["exit", "quit", "thoát"]:
                break
            
            print("🤖 Chatbot đang suy nghĩ...\n")
            
            result = run_pipeline(
                query=query,
                qdrant_store=qdrant,
                bm25_data=bm25_data,
                embedder=embedder,
            )

            print(f"{result.answer}\n")
            print(f"[⏱️ Latency: Tổng {result.latency_breakdown.get('total', 0):.2f}s "
                  f"(Rewrite: {result.latency_breakdown.get('rewrite', 0):.2f}s, "
                  f"Retrieval: {result.latency_breakdown.get('retrieval', 0):.2f}s, "
                  f"Generate: {result.latency_breakdown.get('generation', 0):.2f}s)]")
            
            if result.sources_used:
                print("\n[📚 Nguồn tham khảo]")
                for i, source in enumerate(result.sources_used):
                    print(f"  {i+1}. {source.source} (§ {source.section or 'N/A'})")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")

if __name__ == "__main__":
    main()
