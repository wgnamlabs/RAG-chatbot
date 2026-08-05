from dataclasses import dataclass


@dataclass
class EmbedderConfig:
    model_name: str
    batch_size: int = 32
    max_seq_length: int = 512
    device: str = "cuda"
    trust_remote_code: bool = False
    note: str = ""


# ---------------------------------------------------------------------------
# Danh sách 3 model so sánh (cùng hạng cân ~560M tham số):
#
#   bge-m3                  : Baseline đa ngôn ngữ mạnh, max_seq=8192 token.
#
#   multilingual-e5-large   : 560M — cùng tầm bge-m3, top MTEB multilingual.
#                             Cần prefix "query: " / "passage: " (xem embedder.py).
#                             max_seq=512 (giới hạn kiến trúc gốc của model).
#                             ⚠ So sánh "as-designed": bge-m3/ViEmbed xử lý
#                               chunk dài hơn → không phải so kiến trúc thuần túy.
#
#   Vietnamese_Embedding    : Fine-tune từ bge-m3 trên dữ liệu tiếng Việt.
#                             Đại diện nhóm "chuyên biệt tiếng Việt", max_seq=8192.
# ---------------------------------------------------------------------------
MODELS_TO_COMPARE = [
    EmbedderConfig(
        model_name="BAAI/bge-m3",
        batch_size=32,
        max_seq_length=1024,  # <- đây giờ là mức sàn, quyết định max_chunk_tokens ở trên
        device="cuda",
        note="Baseline đa ngôn ngữ, không chuyên Việt",
    ),
    EmbedderConfig(
        model_name="AITeamVN/Vietnamese_Embedding",
        batch_size=32,
        max_seq_length=2048,
        device="cuda",
        note="Model chuyên Việt, kết quả tốt nhất trong benchmark trước",
    ),
    EmbedderConfig(
        model_name="dangvantuan/vietnamese-document-embedding",
        batch_size=16,
        max_seq_length=8192,
        device="cuda",
        trust_remote_code=True,
        note="Model mới, context dài, cùng họ với model dùng để chunk",
    ),
]