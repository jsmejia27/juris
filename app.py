# app.py
import streamlit as st
import time
from typing import List, Dict, Any
from rag_pipeline import LegalRAGPipeline, DEFAULT_COLLECTION, DEFAULT_STORAGE_DIR

# Page Configuration
st.set_page_config(
    page_title="Juris - Philippine Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished legal UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .source-badge {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rag_pipeline() -> LegalRAGPipeline:
    return LegalRAGPipeline(
        llm_model="qwen3.5:9b",
        temperature=0.1
    )

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scales--v1.png", width=64)
    st.title("⚖️ Juris AI Settings")
    st.caption("Philippine Legal Research Assistant")
    
    st.divider()
    
    # Model Selection
    available_models = [
        "qwen3.5:9b",
        "qwen3:14b",
        "llama3.2:latest",
        "phi4-mini-reasoning:latest",
        "qwen3.8:latest",
        "qwen2.5vl:latest"
    ]
    selected_model = st.selectbox("LLM Model (Local Ollama)", available_models, index=0)
    temperature = st.slider("Temperature (Creativity / Precision)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    
    st.markdown("🚀 **GPU Mode (RTX 5070 Ti - 16GB VRAM)**")
    num_ctx_options = [4096, 8192, 16384]
    selected_num_ctx = st.selectbox("Context Window (num_ctx)", num_ctx_options, index=1, help="Larger context window allows fitting multiple comprehensive legal statutes and rulings in VRAM.")

    st.divider()
    
    # Retrieval Filters
    st.subheader("🔍 Retrieval Filters")
    category_filter = st.selectbox("Document Category", ["All", "Republic Act", "Jurisprudence"], index=0)
    
    col_y1, col_y2 = st.columns(2)
    with col_y1:
        year_min = st.number_input("From Year", min_value=1901, max_value=2026, value=1901)
    with col_y2:
        year_max = st.number_input("To Year", min_value=1901, max_value=2026, value=2026)
        
    top_k = st.slider("Top Sources to Retrieve (Top-K)", min_value=1, max_value=8, value=4)
    
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Header & Disclaimer
st.markdown('<div class="main-header">⚖️ Juris: Philippine Legal AI Research</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Retrieval-Augmented Generation over Supreme Court Jurisprudence & Philippine Statutes</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <strong>LEGAL DISCLAIMER:</strong> This application is an AI research aid powered by local embeddings and models. It is designed to assist in locating relevant Philippine statutes, Republic Acts, and Supreme Court rulings. It does <strong>not</strong> provide formal legal advice and is not a substitute for a licensed Philippine attorney.
</div>
""", unsafe_allow_html=True)

# Initialize RAG Pipeline
try:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = get_rag_pipeline()
    pipeline = st.session_state.pipeline
    pipeline.set_model(selected_model)
    pipeline.set_temperature(temperature)
    pipeline.set_num_ctx(selected_num_ctx)
except Exception as e:
    st.error(f"Failed to initialize RAG pipeline. Make sure Ollama is running at `http://127.0.0.1:11434`. Error: {e}")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am **Juris**, your Philippine Legal AI Research Assistant. You can ask me questions about Philippine statutes (Republic Acts) or Supreme Court jurisprudence (G.R. rulings). How can I assist your legal research today?",
            "sources": []
        }
    ]

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display Sources if available
        if msg.get("sources"):
            with st.expander(f"📚 View Retrieved Legal Sources ({len(msg['sources'])} citations)"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"""
                    **{i}. [{src.get('category', 'Document')}] {src.get('title', 'Unknown Title')}**  
                    *Citation:* `{src.get('gr_no', 'N/A')}` | *Year/Date:* `{src.get('date') or src.get('year') or 'N/A'}` | *Relevance Score:* `{src.get('score', 0.0):.4f}`
                    """)
                    if src.get("ponente"):
                        st.caption(f"**Ponente / Author:** {src['ponente']}")
                    if src.get("summary"):
                        st.info(f"**Summary:** {src['summary']}")
                    st.text_area(f"Excerpt (Chunk {src.get('chunk_index', 1)}/{src.get('total_chunks', 1)})", src.get("text", ""), height=120, key=f"hist_{msg['content'][:10]}_{i}")
                    st.markdown("---")

# User Input
if user_input := st.chat_input("Enter your legal query (e.g. 'What are the penalties under RA 10?' or 'Elements of estafa')"):
    # Add User Message to History
    st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching Philippine legal corpus & analyzing with local LLM..."):
            # 1. Retrieve Sources
            retrieved_sources = pipeline.retriever.retrieve(
                query=user_input,
                limit=top_k,
                category=category_filter,
                year_min=int(year_min),
                year_max=int(year_max)
            )
            
            # 2. Format Context & Prompt
            context_str = pipeline.format_context(retrieved_sources)
            from rag_pipeline import SYSTEM_PROMPT_TEMPLATE
            prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_str, question=user_input)
            
            # 3. Stream Response
            def generate_stream():
                for chunk in pipeline.llm.stream(prompt):
                    yield chunk

            response_container = st.empty()
            full_response = ""
            for chunk in generate_stream():
                full_response += chunk
                response_container.markdown(full_response + "▌")
            response_container.markdown(full_response)
            
            # 4. Display Sources Expander
            if retrieved_sources:
                with st.expander(f"📚 View Retrieved Legal Sources ({len(retrieved_sources)} citations)"):
                    for i, src in enumerate(retrieved_sources, 1):
                        st.markdown(f"""
                        **{i}. [{src.get('category', 'Document')}] {src.get('title', 'Unknown Title')}**  
                        *Citation:* `{src.get('gr_no', 'N/A')}` | *Year/Date:* `{src.get('date') or src.get('year') or 'N/A'}` | *Relevance Score:* `{src.get('score', 0.0):.4f}`
                        """)
                        if src.get("ponente"):
                            st.caption(f"**Ponente / Author:** {src['ponente']}")
                        if src.get("summary"):
                            st.info(f"**Summary:** {src['summary']}")
                        st.text_area(f"Excerpt (Chunk {src.get('chunk_index', 1)}/{src.get('total_chunks', 1)})", src.get("text", ""), height=120, key=f"curr_{i}_{time.time()}")
                        st.markdown("---")

        # Save Assistant Message to History
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": retrieved_sources
        })
