"""SentenceTransformer embedder dùng cho benchmark embedding.

Điểm quan trọng:
- AITeamVN/Vietnamese_Embedding_v2: dùng text trực tiếp.
- BAAI/bge-m3: dùng text trực tiếp.
- Qwen/Qwen3-Embedding-4B:
    * query: dùng prompt_name="query"
    * document/chunk: không dùng query prompt
- Auto-batch chỉ giảm batch size khi CUDA OOM, không âm thầm đổi model.
"""

import gc
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .base import BaseEmbedder
from .config import EmbedderConfig


def _is_qwen3_embedding(model_name: str) -> bool:
    return "qwen3-embedding" in model_name.lower()


def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


def _log_gpu_memory(tag: str = "") -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    prefix = f"[{tag}] " if tag else ""
    print(f"{prefix}GPU memory: {allocated:.2f}GB / {total:.2f}GB")


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, config: EmbedderConfig):
        self.config = config
        self.model = None
        self.runtime_device = config.device

    def load(self) -> None:
        if self.model is not None:
            return

        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[Embedder] CUDA không khả dụng, chuyển sang CPU.")
            device = "cpu"

        self.runtime_device = device
        print(f"[Embedder] Loading {self.config.model_name} -> {device} ...")

        self.model = SentenceTransformer(
            self.config.model_name,
            device=device,
            trust_remote_code=self.config.trust_remote_code,
        )

        effective_max = getattr(self.model, "max_seq_length", None)
        if effective_max is not None:
            print(
                f"[Embedder] model.max_seq_length={effective_max}; "
                f"benchmark configured budget={self.config.max_seq_length}"
            )

        if _is_qwen3_embedding(self.config.model_name):
            prompts = getattr(self.model, "prompts", {}) or {}
            if "query" not in prompts:
                raise RuntimeError(
                    "Qwen3-Embedding được benchmark với prompt_name='query', "
                    "nhưng SentenceTransformer model hiện không expose prompt 'query'. "
                    "Hãy cập nhật sentence-transformers/model snapshot thay vì "
                    "âm thầm chạy Qwen không instruction."
                )

        _log_gpu_memory(f"Loaded {self.config.model_name}")

    def unload(self) -> None:
        if self.model is None:
            return
        print(f"[Embedder] Unloading {self.config.model_name}...")
        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _preprocess(self, texts: List[str], is_query: bool) -> List[str]:
        if _is_e5(self.config.model_name):
            prefix = _E5_QUERY_PREFIX if is_query else _E5_PASSAGE_PREFIX
            return [prefix + t for t in texts]
        return texts

    def _encode_kwargs(self, is_query: bool, batch_size: int) -> Dict:
        kwargs = {
            "batch_size": batch_size,
            "convert_to_numpy": True,
            "show_progress_bar": True,
        }
        if _is_qwen3_embedding(self.config.model_name) and is_query:
            kwargs["prompt_name"] = "query"
        return kwargs

    def _encode_with_auto_batch(
        self,
        processed: List[str],
        is_query: bool,
    ) -> np.ndarray:
        batch_size = self.config.batch_size
        min_batch_size = self.config.min_batch_size

        while batch_size >= min_batch_size:
            try:
                result = self.model.encode(
                    processed,
                    **self._encode_kwargs(is_query, batch_size),
                )
                _log_gpu_memory(f"After encode (batch_size={batch_size})")
                return result
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                new_batch_size = batch_size // 2
                print(
                    f"[Embedder][OOM] batch_size={batch_size} vượt VRAM, "
                    f"giảm xuống {new_batch_size} và thử lại..."
                )
                batch_size = new_batch_size

        raise RuntimeError(
            f"[Embedder] Không thể encode {self.config.model_name} "
            f"dù đã giảm batch_size xuống {min_batch_size}."
        )

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if self.model is None:
            self.load()

        processed = self._preprocess(texts, is_query)
        print(
            f"[Embedder] Encoding {len(texts)} texts "
            f"(is_query={is_query}, batch_size={self.config.batch_size}, "
            f"auto_batch={self.config.auto_batch})..."
        )

        if self.config.auto_batch and self.runtime_device == "cuda":
            return self._encode_with_auto_batch(processed, is_query=is_query)

        return self.model.encode(
            processed,
            **self._encode_kwargs(is_query, self.config.batch_size),
        )
