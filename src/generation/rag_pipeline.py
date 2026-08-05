"""
rag_pipeline.py — Pipeline RAG hoàn chỉnh cho chatbot phụ sản.

Luồng xử lý:
  query
    → [InputGuardrail]       — chặn câu hỏi chẩn đoán cá nhân / OOD
    → [QueryRewriter]        — mở rộng viết tắt, paraphrase (qwen2.5:7b)
    → [HybridRetriever]      — Dense (Qdrant) + BM25, merge RRF
    → [CrossEncoderReranker] — bge-reranker-v2-m3, top-5
    → [PromptBuilder]        — citation template, "không biết" instruction
    → [OllamaLLMClient]      — qwen2.5:7b generation
    → [OutputGuardrail]      — faithfulness check
    → response: {answer, sources, rewritten_query, is_faithful}

Lazy loading: các model được load theo yêu cầu, không load tất cả khi khởi tạo.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Generator


@dataclass
class RAGConfig:
    """Cấu hình toàn bộ RAG pipeline.

    Attributes:
        base_path:        Đường dẫn gốc project (Path object).
        embedding_model:  HuggingFace embedding model.
        embedding_dim:    Số chiều embedding.
        embedding_device: "cuda" | "cpu".
        reranker_top_k:   Số kết quả sau rerank.
        candidate_k:      Số candidates lấy từ hybrid trước khi rerank.
        llm_model:        Tên model Ollama cho generation.
        rewriter_model:   Tên model Ollama cho query rewriting (mặc định = llm_model).
        enable_rewrite:   Bật/tắt query rewriting.
        enable_rerank:    Bật/tắt reranking.
        enable_guardrails:Bật/tắt guardrails.
        stream:           Bật streaming cho generation.
        ollama_url:       URL Ollama server.
    """
    base_path: Path = field(default_factory=lambda: Path("."))
    embedding_model: str = "AITeamVN/Vietnamese_Embedding"
    embedding_dim: int = 1024
    embedding_device: str = "cuda"
    reranker_top_k: int = 5
    candidate_k: int = 50
    llm_model: str = "qwen2.5:7b"
    rewriter_model: str = ""  # Nếu rỗng, dùng llm_model
    enable_rewrite: bool = True
    enable_rerank: bool = True
    enable_guardrails: bool = True
    stream: bool = False
    ollama_url: str = "http://localhost:11434"

    def __post_init__(self):
        if not self.rewriter_model:
            self.rewriter_model = self.llm_model


@dataclass
class RAGResponse:
    """Kết quả trả về từ RAG pipeline.

    Attributes:
        answer:           Câu trả lời cuối cùng (đã thêm disclaimer nếu cần).
        sources:          Danh sách source trích dẫn [{index, source_file, chunk_id}].
        rewritten_query:  Câu hỏi sau khi rewrite (giống gốc nếu rewrite tắt).
        is_faithful:      Kết quả faithfulness check (None nếu không check).
        blocked:          True nếu câu hỏi bị guardrail chặn.
        pipeline_stages:  Thông tin debug về từng stage.
    """
    answer: str
    sources: List[dict]
    rewritten_query: str
    is_faithful: Optional[bool] = None
    blocked: bool = False
    pipeline_stages: dict = field(default_factory=dict)


class RAGPipeline:
    """Pipeline RAG hoàn chỉnh cho chatbot phụ sản.

    Ví dụ:
        from pathlib import Path
        pipeline = RAGPipeline.from_base_path(Path("d:/rag-phu-san-chatbot"))
        response = pipeline.run("đái tháo đường thai kỳ có nguy hiểm không?")
        print(response.answer)
        print(response.sources)
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._retriever = None
        self._reranker  = None
        self._rewriter  = None
        self._llm       = None
        self._input_guard  = None
        self._output_guard = None

    @classmethod
    def from_base_path(cls, base_path: Path, **kwargs) -> "RAGPipeline":
        """Tạo pipeline từ đường dẫn gốc project."""
        config = RAGConfig(base_path=base_path, **kwargs)
        return cls(config)

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _get_retriever(self):
        if self._retriever is not None:
            return self._retriever

        import sys
        sys.path.insert(0, str(self.config.base_path / "src"))

        from indexing.embedding import SentenceTransformerEmbedder
        from indexing.embedding.config import EmbedderConfig
        from indexing.vector_store import QdrantVectorStore, QdrantStoreConfig
        from generation.retrieval.bm25_retriever import BM25Retriever
        from generation.retrieval.dense_retriever import DenseRetriever
        from generation.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig

        bm25_pickle = self.config.base_path / "data" / "vector_db" / "bm25_index.pkl"
        if not bm25_pickle.exists():
            raise FileNotFoundError(
                f"BM25 index không tồn tại: {bm25_pickle}\n"
                "Hãy chạy: python evaluation/build_vector_store.py"
            )

        bm25 = BM25Retriever()
        bm25.load_from_pickle(bm25_pickle)

        emb_config = EmbedderConfig(
            model_name=self.config.embedding_model,
            batch_size=32,
            max_seq_length=4096,
            device=self.config.embedding_device,
        )
        embedder = SentenceTransformerEmbedder(emb_config)
        embedder.load()

        qdrant_config = QdrantStoreConfig(
            host="localhost",
            port=6333,
            vector_size=self.config.embedding_dim,
        )
        store = QdrantVectorStore(config=qdrant_config)
        store.load()

        dense = DenseRetriever(store=store, embedder=embedder)
        self._retriever = HybridRetriever(
            dense, bm25, HybridRetrieverConfig(rrf_k=60)
        )
        return self._retriever

    def _get_reranker(self):
        if self._reranker is not None:
            return self._reranker
        from generation.retrieval.reranker import CrossEncoderReranker, RerankerConfig
        self._reranker = CrossEncoderReranker(RerankerConfig(
            model_name="BAAI/bge-reranker-v2-m3",
            device="auto",
        ))
        self._reranker.load()
        return self._reranker

    def _get_rewriter(self):
        if self._rewriter is not None:
            return self._rewriter
        from generation.pre_retrieval.query_rewriter import QueryRewriter, QueryRewriterConfig
        self._rewriter = QueryRewriter(QueryRewriterConfig(
            model=self.config.rewriter_model,
            base_url=self.config.ollama_url,
        ))
        return self._rewriter

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        from generation.llm_client import OllamaLLMClient, LLMConfig
        self._llm = OllamaLLMClient(LLMConfig(
            model=self.config.llm_model,
            base_url=self.config.ollama_url,
            temperature=0.0,
            max_tokens=1024,
            stream=self.config.stream,
        ))
        return self._llm

    def _get_guardrails(self):
        if self._input_guard is None:
            from generation.guardrails import InputGuardrail, OutputGuardrail
            self._input_guard  = InputGuardrail()
            self._output_guard = OutputGuardrail(llm_client=self._get_llm())
        return self._input_guard, self._output_guard

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, query: str) -> RAGResponse:
        """Chạy full RAG pipeline.

        Args:
            query: Câu hỏi từ người dùng.

        Returns:
            RAGResponse với answer, sources, và metadata pipeline.
        """
        from generation.prompt_builder import build_prompt, add_disclaimer_if_needed

        stages = {}

        # 1. Input guardrail
        if self.config.enable_guardrails:
            input_guard, _ = self._get_guardrails()
            blocked_response = input_guard.check(query)
            if blocked_response:
                return RAGResponse(
                    answer=blocked_response,
                    sources=[],
                    rewritten_query=query,
                    is_faithful=True,
                    blocked=True,
                    pipeline_stages={"blocked": True, "reason": "input_guardrail"},
                )
        stages["input_guardrail"] = "passed"

        # 2. Query rewriting
        rewritten_query = query
        if self.config.enable_rewrite:
            rewritten_query = self._get_rewriter().rewrite(query)
            stages["rewriter"] = rewritten_query

        # 3. Hybrid retrieval
        retriever = self._get_retriever()
        candidates = retriever.retrieve(rewritten_query, top_k=self.config.candidate_k)
        stages["retrieval_count"] = len(candidates)

        # 4. Rerank
        final_results = candidates
        if self.config.enable_rerank and candidates:
            reranker = self._get_reranker()
            final_results = reranker.rerank(
                rewritten_query, candidates, top_k=self.config.reranker_top_k
            )
        stages["rerank_count"] = len(final_results)

        # 5. Build prompt
        contexts = [
            {"text": r.text, "metadata": r.metadata, "score": r.score}
            for r in final_results
        ]
        messages, sources = build_prompt(rewritten_query, contexts)
        stages["context_count"] = len(contexts)

        # 6. Generate
        llm = self._get_llm()
        answer = llm.chat(messages)

        # Thêm disclaimer nếu cần
        answer = add_disclaimer_if_needed(answer)

        # 7. Output guardrail (faithfulness check)
        is_faithful = None
        if self.config.enable_guardrails:
            _, output_guard = self._get_guardrails()
            answer, is_faithful = output_guard.check_faithfulness(answer, contexts)
        stages["is_faithful"] = is_faithful

        return RAGResponse(
            answer=answer,
            sources=sources,
            rewritten_query=rewritten_query,
            is_faithful=is_faithful,
            blocked=False,
            pipeline_stages=stages,
        )

    def stream(self, query: str) -> Generator[str, None, None]:
        """Streaming version — yield từng token, trả về sources cuối.

        Dùng cho Streamlit UI với st.write_stream().
        Guardrails vẫn chạy nhưng output không stream (instant response).

        Yields:
            Từng chunk text từ LLM.
        """
        from generation.prompt_builder import build_prompt, add_disclaimer_if_needed

        # Input guardrail
        if self.config.enable_guardrails:
            input_guard, _ = self._get_guardrails()
            blocked = input_guard.check(query)
            if blocked:
                yield blocked
                return

        # Rewrite
        rewritten = self._get_rewriter().rewrite(query) if self.config.enable_rewrite else query

        # Retrieve & Rerank
        candidates = self._get_retriever().retrieve(rewritten, top_k=self.config.candidate_k)
        final_results = (
            self._get_reranker().rerank(rewritten, candidates, top_k=self.config.reranker_top_k)
            if self.config.enable_rerank else candidates
        )

        contexts = [
            {"text": r.text, "metadata": r.metadata, "score": r.score}
            for r in final_results
        ]
        messages, _ = build_prompt(rewritten, contexts)

        # Stream generation
        llm = self._get_llm()
        for chunk in llm.chat_stream(messages):
            yield chunk
