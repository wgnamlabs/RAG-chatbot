"""
run_ragas_eval.py — End-to-end evaluation với RAGAS metrics.

Metrics:
  - faithfulness:       Câu trả lời có trung thực với context không?
  - answer_relevancy:   Câu trả lời có trả lời đúng câu hỏi không?
  - context_precision:  Context retrieve được có liên quan đến câu hỏi không?

LLM judge: qwen2.5:7b qua Ollama (không dùng OpenAI API).

Chạy:
    python evaluation/run_ragas_eval.py
    python evaluation/run_ragas_eval.py --device cpu --n_questions 10
    python evaluation/run_ragas_eval.py --no_rewrite --no_rerank  # pipeline đơn giản

Output: evaluation/results/ragas_scores.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

# Reconfigure stdout for Windows console
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def build_pipeline(base_path: Path, device: str, enable_rewrite: bool, enable_rerank: bool):
    """Tạo RAG pipeline với config chỉ định."""
    from generation.rag_pipeline import RAGPipeline, RAGConfig
    config = RAGConfig(
        base_path=base_path,
        embedding_device=device,
        enable_rewrite=enable_rewrite,
        enable_rerank=enable_rerank,
        enable_guardrails=False,  # Tắt guardrails khi eval để không filter câu hỏi
    )
    return RAGPipeline(config)


def run_ragas_eval(
    base_path: Path,
    device: str = "cuda",
    n_questions: int = None,
    enable_rewrite: bool = True,
    enable_rerank: bool = True,
) -> None:
    output_csv = base_path / "evaluation" / "results" / "ragas_scores.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── Load questions (in-domain only) ────────────────────────────────────────
    questions_path = base_path / "evaluation" / "eval_questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # Chỉ dùng in-domain (có ground truth)
    in_domain = [item for item in qa_data if item["ground_truth_sources"]]
    if n_questions:
        in_domain = in_domain[:n_questions]
    print(f"[RAGAS] Đánh giá {len(in_domain)} câu hỏi in-domain...")

    # ── Build pipeline ──────────────────────────────────────────────────────────
    pipeline = build_pipeline(base_path, device, enable_rewrite, enable_rerank)
    pipeline_name = (
        f"Hybrid+Rerank" if enable_rewrite and enable_rerank else
        f"Hybrid+Rewrite" if enable_rewrite else
        f"Hybrid+Rerank(no rewrite)" if enable_rerank else
        f"Hybrid only"
    )

    # ── Collect pipeline outputs ────────────────────────────────────────────────
    questions_list   = []
    answers_list     = []
    contexts_list    = []  # list of list[str]

    for i, item in enumerate(in_domain):
        q = item["question"]
        print(f"  [{i+1}/{len(in_domain)}] {q[:60]}...")
        try:
            response = pipeline.run(q)
            questions_list.append(q)
            answers_list.append(response.answer)
            contexts_list.append([ctx["text"] for ctx in [
                {"text": c.get("text", ""), "metadata": c.get("metadata", {})}
                for c in []  # placeholder
            ]])
        except Exception as e:
            print(f"    ⚠️  Lỗi: {e}")
            questions_list.append(q)
            answers_list.append("(ERROR)")
            contexts_list.append([""])

    # Collect contexts separately (pipeline doesn't return them directly in run())
    # Re-run with context extraction
    questions_list   = []
    answers_list     = []
    contexts_list    = []
    references_list  = []

    from generation.retrieval.bm25_retriever import BM25Retriever
    from generation.retrieval.dense_retriever import DenseRetriever
    from generation.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig
    from generation.retrieval.reranker import CrossEncoderReranker, RerankerConfig
    from generation.pre_retrieval.query_rewriter import QueryRewriter, QueryRewriterConfig
    from generation.prompt_builder import build_prompt
    from generation.llm_client import OllamaLLMClient, LLMConfig
    from indexing.embedding import SentenceTransformerEmbedder
    from indexing.embedding.config import EmbedderConfig
    from indexing.vector_store import QdrantVectorStore, QdrantStoreConfig

    # Load components
    bm25_pickle = base_path / "data" / "vector_db" / "bm25_index.pkl"
    bm25 = BM25Retriever()
    bm25.load_from_pickle(bm25_pickle)

    emb_config = EmbedderConfig(
        model_name="AITeamVN/Vietnamese_Embedding",
        batch_size=32, max_seq_length=4096, device=device,
    )
    embedder = SentenceTransformerEmbedder(emb_config)
    embedder.load()

    qdrant_config = QdrantStoreConfig(
        host="localhost", port=6333,
    )
    store = QdrantVectorStore(config=qdrant_config)
    store.load()

    dense    = DenseRetriever(store=store, embedder=embedder)
    hybrid   = HybridRetriever(dense, bm25, HybridRetrieverConfig(rrf_k=60))
    reranker = CrossEncoderReranker(RerankerConfig()) if enable_rerank else None
    if reranker:
        reranker.load()

    rewriter = QueryRewriter() if enable_rewrite else None

    llm = OllamaLLMClient(LLMConfig(model="qwen2.5:7b", temperature=0.0))

    for i, item in enumerate(in_domain):
        q = item["question"]
        print(f"  [{i+1}/{len(in_domain)}] {q[:60]}...")
        try:
            # Rewrite
            rq = rewriter.rewrite(q) if rewriter else q

            # Retrieve + Rerank
            candidates = hybrid.retrieve(rq, top_k=20)
            final = (reranker.rerank(rq, candidates, top_k=5)
                     if reranker else candidates[:5])

            contexts = [r.text for r in final]
            ctx_dicts = [{"text": r.text, "metadata": r.metadata} for r in final]

            # Generate
            messages, _ = build_prompt(rq, ctx_dicts)
            answer = llm.chat(messages)

            questions_list.append(q)
            answers_list.append(answer)
            contexts_list.append(contexts)
            references_list.append(item.get("ground_truth") or "")
        except Exception as e:
            print(f"    ⚠️  Lỗi: {e}")
            questions_list.append(q)
            answers_list.append("(ERROR)")
            contexts_list.append([""])
            references_list.append(item.get("ground_truth") or "")

    if reranker:
        reranker.unload()

    # ── RAGAS evaluation ────────────────────────────────────────────────────────
    print(f"\n[RAGAS] Tính metrics với qwen2.5:7b làm judge...")

    try:
        # Workaround cho lỗi: No module named 'langchain_community.chat_models.vertexai'
        # do Langchain >= 0.3 xoá API, RAGAS 0.3.x vẫn cố import
        import sys
        import unittest.mock
        sys.modules["langchain_community.chat_models.vertexai"] = unittest.mock.MagicMock()
        sys.modules["langchain_community.embeddings.vertexai"] = unittest.mock.MagicMock()
        sys.modules["langchain_community.llms.vertexai"] = unittest.mock.MagicMock()

        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from datasets import Dataset

        # Setup RAGAS với Ollama
        ollama_llm = ChatOllama(model="qwen2.5:7b", temperature=0)
        ollama_emb = OllamaEmbeddings(model="nomic-embed-text")
        ragas_llm  = LangchainLLMWrapper(ollama_llm)
        ragas_emb  = LangchainEmbeddingsWrapper(ollama_emb)

        # Prepare dataset
        dataset = Dataset.from_dict({
            "question":  questions_list,
            "answer":    answers_list,
            "contexts":  contexts_list,
            "reference": references_list,
        })

        # Run evaluation
        metrics = [faithfulness, answer_relevancy, context_precision]
        for metric in metrics:
            metric.llm = ragas_llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = ragas_emb

        run_config = RunConfig(timeout=300, max_workers=2, max_retries=3)
        result = evaluate(dataset, metrics=metrics, run_config=run_config)
        df = result.to_pandas()

        # Summary
        summary = {
            "pipeline":         pipeline_name,
            "n_questions":      len(in_domain),
            "faithfulness":     round(df["faithfulness"].mean(), 4),
            "answer_relevancy": round(df["answer_relevancy"].mean(), 4),
            "context_precision": round(df["context_precision"].mean(), 4),
        }

        print(f"\n{'='*55}")
        print(f"RAGAS Summary — {pipeline_name}")
        print(f"{'='*55}")
        for k, v in summary.items():
            if k not in ("pipeline", "n_questions"):
                print(f"  {k:<25}: {v}")
        print(f"{'='*55}\n")

        # Save CSV
        fieldnames = list(summary.keys())
        existing_rows = []
        if output_csv.exists():
            with open(output_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                existing_rows = [r for r in reader if r.get("pipeline") != pipeline_name]

        all_rows = existing_rows + [summary]
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        # Per-question CSV
        per_q_path = output_csv.parent / "ragas_per_question.csv"
        df.insert(0, "pipeline", pipeline_name)
        df.to_csv(per_q_path, index=False, encoding="utf-8")

        print(f"✅ RAGAS scores → {output_csv}")
        print(f"   Per-question → {per_q_path}")

    except ImportError as e:
        print(f"❌ RAGAS không được cài: {e}")
        print("   Hãy chạy: pip install ragas langchain-ollama datasets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--n_questions", default=None, type=int,
                        help="Số câu hỏi để test (None = tất cả in-domain)")
    parser.add_argument("--no_rewrite",  action="store_true")
    parser.add_argument("--no_rerank",   action="store_true")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent
    run_ragas_eval(
        base_path,
        device=args.device,
        n_questions=args.n_questions,
        enable_rewrite=not args.no_rewrite,
        enable_rerank=not args.no_rerank,
    )
