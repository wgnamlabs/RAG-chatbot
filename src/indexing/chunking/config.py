from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional


DEFAULT_TOKENIZER_MODEL = "AITeamVN/Vietnamese_Embedding_v2"


def default_markdown_headers() -> List[tuple]:
    """Giữ toàn bộ hierarchy H1-H6 của corpus đã clean."""
    return [
        ("#",      "Header 1"),
        ("##",     "Header 2"),
        ("###",    "Header 3"),
        ("####",   "Header 4"),
        ("#####",  "Header 5"),
        ("######", "Header 6"),
    ]


@dataclass
class SemanticChunkerConfig:
    embedding_model_name: str = DEFAULT_TOKENIZER_MODEL
    trust_remote_code: bool = False

    headers_to_split_on: List[tuple] = field(default_factory=default_markdown_headers)

    breakpoint_threshold_type: Literal[
        "percentile", "standard_deviation", "interquartile", "gradient"
    ] = "percentile"
    breakpoint_threshold_amount: float = 95.0
    buffer_size: int = 1

    # Size guard dùng TOKEN thật của chính embedding model.
    min_chunk_tokens: int = 30
    max_chunk_tokens: int = 1024
    overflow_overlap_tokens: int = 50

    # Bảng lớn được tách theo row, mỗi part lặp header + separator.
    table_row_overlap: int = 0

    # Trong evaluation không được âm thầm fallback vì sẽ làm sai kết luận.
    # Có thể đặt False ở production nếu muốn ưu tiên robustness.
    raise_on_semantic_error: bool = True

    def __post_init__(self):
        assert 0 < self.min_chunk_tokens < self.max_chunk_tokens
        assert 0 <= self.overflow_overlap_tokens < self.max_chunk_tokens
        assert self.table_row_overlap >= 0


@dataclass
class HierarchicalChunkerConfig:
    tokenizer_model_name: str = DEFAULT_TOKENIZER_MODEL
    trust_remote_code: bool = False

    headers_to_split_on: List[tuple] = field(default_factory=default_markdown_headers)

    # Hai kỹ thuật dùng cùng một max token budget để so sánh công bằng.
    # Tên field cũ được giữ để tránh làm vỡ code caller hiện tại.
    child_chunk_size: int = 1024
    child_chunk_overlap: int = 100

    # Bảng lớn được tách theo row, mỗi part lặp header + separator.
    table_row_overlap: int = 0

    # Optional cho trường hợp caller muốn custom splitter metric.
    # Mặc định None => dùng tokenizer thật, KHÔNG dùng len() ký tự nữa.
    length_function: Optional[Callable[[str], int]] = None

    def __post_init__(self):
        assert self.child_chunk_size > 0
        assert 0 <= self.child_chunk_overlap < self.child_chunk_size
        assert self.table_row_overlap >= 0
