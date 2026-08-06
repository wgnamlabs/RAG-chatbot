from dataclasses import dataclass


@dataclass
class EmbedderConfig:
    model_name: str
    batch_size: int = 32
    max_seq_length: int = 512
    device: str = "cuda"
    trust_remote_code: bool = False
    note: str = ""

    # ------------------------------------------------------------------
    # Auto-batch: nếu True, encoder sẽ TỰ ĐỘNG lùi batch_size khi gặp
    # CUDA OutOfMemoryError, bắt đầu từ batch_size ở trên rồi giảm dần
    # cho tới min_batch_size. Bật cho các model nặng (4B/8B) để tận dụng
    # tối đa VRAM còn dư mà không lo crash giữa chừng trên Colab.
    # ------------------------------------------------------------------
    auto_batch: bool = False
    min_batch_size: int = 1


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
        # Tăng từ 4 lên 16: log thực tế cho thấy VRAM còn dư ~5GB ở batch=4
        # (9.7/15.0GB đã dùng). Bật auto_batch để tự lùi 16 -> 8 -> 4 -> 2
        # nếu gặp OOM (do chunk dài không đều trong corpus), tránh phải
        # đoán thủ công và tránh crash giữa chừng.
        batch_size=16,
        max_seq_length=2048,
        device="cuda",
        trust_remote_code=False,  # Qwen3-Embedding không cần custom code
        note="Model lớn, đa ngôn ngữ, thay thế dangvantuan do lỗi CUDA RoPE assert",
        auto_batch=True,
        min_batch_size=2,
    ),
    EmbedderConfig(
        model_name="nvidia/Nemotron-3-Embed-8B-BF16",
        # Model 8B nặng gần gấp đôi Qwen3-4B -> tăng thận trọng hơn, từ 2 lên 6.
        # Bắt buộc bật auto_batch vì model này sát giới hạn VRAM T4 (16GB),
        # rủi ro OOM cao hơn hẳn 2 model còn lại.
        batch_size=6,
        max_seq_length=2048,   # giới hạn để tránh OOM trên T4
        device="cuda",
        note="SOTA đa ngôn ngữ 2026, đại diện 'model mạnh nhất khả thi trên T4'",
        auto_batch=True,
        min_batch_size=1,
    ),
]