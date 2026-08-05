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
        model_name="AITeamVN/Vietnamese_Embedding_v2",
        batch_size=32,
        max_seq_length=2048,
        device="cuda",
        note="Bản nâng cấp trực tiếp của model bạn từng test — so sánh v1 vs v2",
    ),
    EmbedderConfig(
        model_name="Qwen/Qwen3-Embedding-4B",
        batch_size=4,           # giảm mạnh so với 32 vì model 4B, T4 chỉ 16GB VRAM
        max_seq_length=2048,    # đủ dùng, không cần tận dụng hết 32K (tốn VRAM vô ích)
        device="cuda",
        trust_remote_code=False,  # Qwen3-Embedding không cần custom code
        note="Model lớn, đa ngôn ngữ, thay thế dangvantuan do lỗi CUDA RoPE assert",
    ),
    EmbedderConfig(
        model_name="nvidia/Nemotron-3-Embed-8B-BF16",
        batch_size=2,          # cẩn thận VRAM
        max_seq_length=2048,   # giới hạn để tránh OOM trên T4
        device="cuda",
        note="SOTA đa ngôn ngữ 2026, đại diện 'model mạnh nhất khả thi trên T4'",
    ),
]