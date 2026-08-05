from dataclasses import dataclass, field
from typing import Callable, List, Literal


@dataclass
class SemanticChunkerConfig:
    embedding_model_name: str = "AITeamVN/Vietnamese_Embedding_v2"
    trust_remote_code: bool = False

    breakpoint_threshold_type: Literal[
        "percentile", "standard_deviation", "interquartile", "gradient"
    ] = "percentile"
    breakpoint_threshold_amount: float = 95.0
    buffer_size: int = 1

    min_chunk_tokens: int = 30

    # SỬA: e5-large đã bị loại khỏi MODELS_TO_COMPARE, nên mức sàn giờ
    # là bge-m3 (max_seq_length=1024) — model có context nhỏ nhất còn lại.
    # Nâng từ 512 lên 1024 để chunk giữ được ngữ cảnh trọn vẹn hơn.
    max_chunk_tokens: int = 1024

    def __post_init__(self):
        assert self.min_chunk_tokens < self.max_chunk_tokens, (
            "min_chunk_tokens phải nhỏ hơn max_chunk_tokens"
        )


@dataclass
class HierarchicalChunkerConfig:
    headers_to_split_on: List[tuple] = field(
        default_factory=lambda: [
            ("#",    "Header 1"),
            ("##",   "Header 2"),
            ("###",  "Header 3"),
            ("####", "Header 4"),
        ]
    )
    # child_chunk_size đo bằng ĐƠN VỊ của length_function (token nếu dùng
    # tokenizer, ký tự nếu dùng len).  Mặc định đo bằng ký tự để backward-
    # compatible; caller có thể truyền tokenizer_fn để đo bằng token.
    child_chunk_size: int = 1000
    child_chunk_overlap: int = 200

    # Hàm đo độ dài cho RecursiveCharacterTextSplitter.
    # Mặc định: len() (ký tự).
    # Để đo bằng token: truyền lambda text: len(tokenizer.encode(text))
    length_function: Callable[[str], int] = field(default_factory=lambda: len)
