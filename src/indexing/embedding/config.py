from dataclasses import dataclass


@dataclass
class EmbedderConfig:
    model_name: str
    batch_size: int = 32
    max_seq_length: int = 2048
    device: str = "cuda"
    trust_remote_code: bool = False
    note: str = ""
    auto_batch: bool = False
    min_batch_size: int = 1


# 3 model có mục tiêu khác nhau, KHÔNG cùng số tham số:
# - AITeamVN/Vietnamese_Embedding_v2: chuyên biệt tiếng Việt.
# - Qwen/Qwen3-Embedding-4B: model 4B, multilingual, mạnh nhưng nặng.
# - BAAI/bge-m3: baseline multilingual ~568M.
#
# Corpus hiện tại đã được chunk với max 1024 tokens; cả 3 cấu hình bên dưới
# đều đủ khả năng xử lý độ dài đó.
MODELS_TO_COMPARE = [
    EmbedderConfig(
        model_name="AITeamVN/Vietnamese_Embedding_v2",
        batch_size=32,
        max_seq_length=2048,
        device="cuda",
        trust_remote_code=False,
        note="Embedding chuyên biệt tiếng Việt; baseline domain-language.",
    ),
    EmbedderConfig(
        model_name="Qwen/Qwen3-Embedding-4B",
        batch_size=16,
        max_seq_length=2048,
        device="cuda",
        trust_remote_code=False,
        note="Embedding 4B multilingual; query dùng prompt_name='query'.",
        auto_batch=True,
        min_batch_size=2,
    ),
    EmbedderConfig(
        model_name="BAAI/bge-m3",
        batch_size=32,
        max_seq_length=2048,
        device="cuda",
        trust_remote_code=False,
        note="Baseline multilingual mạnh.",
    ),
]
