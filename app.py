"""
app.py — Streamlit UI cho RAG Chatbot Phụ Sản

Chạy:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Đưa src vào path
BASE_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_PATH / "src"))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chatbot Tư Vấn Phụ Sản",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Header */
  .main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 15px rgba(30, 58, 95, 0.3);
  }
  .main-header h1 { margin: 0; font-size: 1.6rem; font-weight: 600; }
  .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.9rem; }

  /* Disclaimer banner */
  .disclaimer-banner {
    background: #fff3cd;
    border-left: 4px solid #f0a500;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
    color: #856404;
  }

  /* Source card */
  .source-card {
    background: #f0f7ff;
    border: 1px solid #cce0f5;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
    color: #1a3c5e;
  }

  /* Rewrite badge */
  .rewrite-badge {
    background: #e8f5e9;
    border: 1px solid #a5d6a7;
    border-radius: 6px;
    padding: 0.4rem 0.7rem;
    font-size: 0.78rem;
    color: #2e7d32;
    margin-bottom: 0.8rem;
  }

  /* Unfaithful warning */
  .unfaithful-warning {
    background: #ffeaea;
    border-left: 4px solid #e53935;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    font-size: 0.82rem;
    color: #b71c1c;
    margin-top: 0.5rem;
  }

  /* Footer */
  .footer {
    text-align: center;
    font-size: 0.75rem;
    color: #888;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #eee;
  }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏥 Chatbot Tư Vấn Sức Khỏe Phụ Sản</h1>
  <p>Hỗ trợ tra cứu thông tin từ tài liệu y khoa phụ sản • Không thay thế tư vấn bác sĩ</p>
</div>
""", unsafe_allow_html=True)

# ── Disclaimer cố định ─────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer-banner">
  ⚠️ <strong>Lưu ý:</strong> Thông tin từ chatbot chỉ mang tính tham khảo từ tài liệu y khoa chính thức.
  Không thay thế chẩn đoán hoặc điều trị của bác sĩ. Mọi quyết định y tế cần được hướng dẫn bởi nhân viên y tế.
</div>
""", unsafe_allow_html=True)


# ── Sidebar: Settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Cài đặt Pipeline")

    enable_rewrite = st.toggle("Query Rewriting (qwen2.5:7b)", value=True,
                               help="Viết lại câu hỏi để cải thiện retrieval")
    enable_rerank  = st.toggle("Reranking (bge-reranker-v2-m3)", value=True,
                               help="Rerank kết quả với cross-encoder")
    enable_guardrails = st.toggle("Guardrails y khoa", value=True,
                                  help="Chặn câu hỏi chẩn đoán cá nhân")
    candidate_k = st.slider("Hybrid candidates", min_value=10, max_value=100, value=50, step=10,
                             help="Số chunks lấy từ Hybrid trước khi rerank (càng cao càng tốt cho Recall)")
    reranker_top_k = st.slider("Reranker top-k", min_value=1, max_value=10, value=5,
                                help="Số chunks sau rerank")

    st.markdown("---")
    st.markdown("### 📚 Nguồn tài liệu")
    st.markdown("""
    - Hướng dẫn Quốc gia ĐTĐ Thai Kỳ
    - Hướng dẫn CSSKSS Quốc gia
    - Thực hành Lâm sàng Sản Phụ Khoa
    """)

    st.markdown("---")
    st.markdown("### 🤖 Mô hình")
    st.markdown("""
    - **Embedding**: Vietnamese_Embedding
    - **Reranker**: bge-reranker-v2-m3
    - **LLM**: qwen2.5:7b (Ollama)
    """)


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None


@st.cache_resource(show_spinner="⏳ Đang khởi tạo pipeline...")
def load_pipeline():
    """Load RAG pipeline (cached)."""
    from generation.rag_pipeline import RAGPipeline, RAGConfig
    config = RAGConfig(
        base_path=BASE_PATH,
        embedding_device="cuda",
        enable_rewrite=True,
        enable_rerank=True,
        enable_guardrails=True,
        candidate_k=50,
        reranker_top_k=5,
        stream=True,
    )
    return RAGPipeline(config)


# ── Chat history display ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Nguồn trích dẫn ({len(msg['sources'])} tài liệu)"):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>[NGUỒN {src["index"]}]</strong> '
                        f'{src["source_file"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input("Nhập câu hỏi về sức khỏe phụ sản...")

if query:
    # Hiển thị câu hỏi
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Load pipeline
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo pipeline: {e}\n\nĐảm bảo đã chạy `build_vector_store.py` và Ollama đang chạy.")
        st.stop()

    # Cập nhật config từ sidebar
    pipeline.config.enable_rewrite    = enable_rewrite
    pipeline.config.enable_rerank     = enable_rerank
    pipeline.config.enable_guardrails = enable_guardrails
    pipeline.config.candidate_k       = candidate_k
    pipeline.config.reranker_top_k    = reranker_top_k

    # Generate response
    with st.chat_message("assistant"):
        try:
            # Run pipeline (non-streaming để có sources và guardrail info)
            pipeline.config.stream = False
            response = pipeline.run(query)

            # Hiển thị rewritten query nếu khác gốc
            if (enable_rewrite
                    and response.rewritten_query
                    and response.rewritten_query != query):
                st.markdown(
                    f'<div class="rewrite-badge">'
                    f'🔄 <strong>Câu hỏi đã viết lại:</strong> {response.rewritten_query}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Câu trả lời
            st.markdown(response.answer)

            # Cảnh báo faithfulness
            if response.is_faithful is False:
                st.markdown(
                    '<div class="unfaithful-warning">'
                    '⚠️ Câu trả lời có thể chứa thông tin ngoài tài liệu gốc. Hãy kiểm tra lại.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            # Sources
            if response.sources and not response.blocked:
                with st.expander(f"📚 Nguồn trích dẫn ({len(response.sources)} tài liệu)"):
                    for src in response.sources:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<strong>[NGUỒN {src["index"]}]</strong> '
                            f'{src["source_file"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # Lưu vào history
            st.session_state.messages.append({
                "role":    "assistant",
                "content": response.answer,
                "sources": response.sources,
            })

        except ConnectionError as e:
            error_msg = (
                f"❌ **Không kết nối được Ollama**\n\n"
                f"Đảm bảo:\n"
                f"1. Ollama đang chạy: `ollama serve`\n"
                f"2. Model đã được pull: `ollama pull qwen2.5:7b`\n\n"
                f"Chi tiết: {e}"
            )
            st.error(error_msg)
        except FileNotFoundError as e:
            st.error(
                f"❌ **Vector store chưa được build**\n\n"
                f"Hãy chạy: `python evaluation/build_vector_store.py`\n\n"
                f"Chi tiết: {e}"
            )
        except Exception as e:
            st.error(f"❌ Lỗi không xác định: {e}")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  RAG Chatbot Phụ Sản • Chạy hoàn toàn local • Không gửi dữ liệu ra ngoài
</div>
""", unsafe_allow_html=True)
