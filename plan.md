# Comprehensive Plan: Philippine Legal AI Chat Application

A complete architecture and stage-by-stage implementation plan for building a high-precision, Retrieval-Augmented Generation (RAG) legal chat application using the local Philippine legal dataset and a locally hosted **Ollama** LLM.

---

## 📊 Environment & Available Assets

### 1. Local LLM & Embedding Engine (Ollama - `localhost:11434`)
- **Available LLMs:**
  - `qwen3:14b` (Primary recommended: strong reasoning and multilingual context)
  - `llama3.2:latest` (Fast, lightweight inference)
  - `phi4-mini-reasoning:latest` (Fast structured reasoning)
  - `qwen3.8:latest`, `qwen2.5vl:latest`
- **Embedding Models:**
  - `nomic-embed-text:latest` (Dense vector embeddings via Ollama)
  - HuggingFace / FastEmbed (Sparse BM25 for exact keyword matching)

### 2. Available Local Datasets (in workspace)
- `juris.parquet` (~704 MB - Supreme Court Jurisprudence / Decisions)
- `repacts.parquet` & `repacts_with_summary.parquet` (~35 MB / ~42 MB - Republic Acts and legislative summaries)
- `consolidated.parquet` (~788 MB - Full consolidated Philippine legal corpus)
- `markdown.tar.gz` (~497 MB - Raw Markdown source documents)

---

## 🏗️ System Architecture & Data Flow

```
[Raw Legal Data (Parquet / Markdown)]
                  │
                  ▼
   [Stage 1: Preprocessing & Parsing]
   ├── Extract Metadata (G.R. No., Title, Date, Ponente, Category)
   └── Chunking (Recursive Splitter with 1000-1500 tokens + overlap)
                  │
                  ▼
   [Stage 2 & 3: Hybrid Indexing in Qdrant]
   ├── Dense Vector Index (Ollama nomic-embed-text / MiniLM)
   ├── Sparse Vector Index (BM25 / SPLADE for exact citations)
   └── Payload Metadata Storage (Year, Court, Doc Type, Source)
                  │
                  ▼
   [Stage 4: Query & Hybrid Retrieval]
   ├── User Query + Filter (e.g. Year range, Law type)
   ├── Reciprocal Rank Fusion (RRF: Dense Semantic + Sparse Keyword)
   └── Top-K Context Selection (Top 3-5 relevant excerpts)
                  │
                  ▼
   [Stage 5: Local Ollama Generation & Guardrails]
   ├── Grounded System Prompt (Strict no-hallucination policy)
   └── Ollama Inference (`qwen3:14b` / `llama3.2`) with Streaming
                  │
                  ▼
   [Stage 6: Streamlit UI]
   ├── Interactive Chat & Streaming Responses
   ├── Expandable Source Documents & Verified Citations
   └── Sidebar Metadata Filters & Mandatory Legal Disclaimers
```

---

## 🚦 Project Stages & Roadmap

### Stage 1: Dataset Inspection & Preprocessing Strategy (COMPLETED)
- [x] Inspect schemas and sample records of `juris.parquet`, `repacts.parquet`, and `consolidated.parquet`.
- [x] Establish standard metadata fields:
  - `doc_id` / `gr_no` (e.g., G.R. No. 123456)
  - `title` (Case title, Parties, or Act name)
  - `date` / `year`
  - `ponente` / `author`
  - `category` / `source` (Jurisprudence, Republic Act, Presidential Decree, etc.)
- [x] Implement text cleaner, index-file filter, and metadata extraction regex.
- [x] Define chunking rules (recursive character chunk size ~1,200 chars, overlap ~200 chars).

### Stage 2: Environment & Dependencies Setup (COMPLETED)
- [x] Create and activate Python virtual environment (`legal_ai_env`).
- [x] Install dependencies:
  - `langchain`, `langchain-community`, `langchain-ollama`, `langchain-text-splitters`
  - `qdrant-client`, `fastembed` (BM25 sparse embedding)
  - `pyarrow`, `pandas`
  - `streamlit`
- [x] Verify local connectivity with Ollama (`http://127.0.0.1:11434`), dense embedding (`nomic-embed-text`, 768-dim), sparse BM25 (`Qdrant/bm25`), and LLMs (`llama3.2`, `qwen3:14b`).

### Stage 3: Hybrid Vector Database (Qdrant) Setup & Ingestion (COMPLETED)
- [x] Configure local Qdrant instance (file storage in `./qdrant_storage`).
- [x] Initialize Qdrant collection `philippine_law` supporting **Hybrid Search**:
  - Dense vector dimension: 768 (`nomic-embed-text` via Ollama).
  - Sparse vector representation: BM25 via FastEmbed (`Qdrant/bm25`).
- [x] Build batch ingestion pipeline ([`ingest_data.py`](file:///c:/Apps/Juris/ingest_data.py)) with progress bars, metadata attachment, contextual chunking, and resumable checkpoints.
- [x] Verified sample batch ingestion of Republic Acts and Jurisprudence decisions.

### Stage 4: Hybrid Retrieval & Context Formulation (COMPLETED)
- [x] Implement Hybrid Retriever combining Dense Semantic Search (`nomic-embed-text`) and Sparse BM25 Search (`Qdrant/bm25`) with Reciprocal Rank Fusion (RRF).
- [x] Apply metadata filters (category, year range, ponente).
- [x] Design strict Philippine legal prompt template with citation requirements and anti-hallucination guardrails.

### Stage 5: Ollama Local LLM Integration (COMPLETED)
- [x] Connect `LangChain` to Ollama local instance (`http://127.0.0.1:11434`).
- [x] Configure LLM models (`qwen3:14b`, `llama3.2:latest`, `phi4-mini-reasoning`).
- [x] Implement streaming token generator and structured RAG query methods in [`rag_pipeline.py`](file:///c:/Apps/Juris/rag_pipeline.py).
- [x] Validated with test queries on Philippine statutes and case law.

### Stage 6: Streamlit Web UI & Source Citation Viewer (COMPLETED)
- [x] Build responsive chat interface in [`app.py`](file:///c:/Apps/Juris/app.py) with streaming token support.
- [x] Add Sidebar Controls:
  - Model selector (`qwen3:14b`, `llama3.2:latest`, `phi4-mini-reasoning:latest`, etc.)
  - Metadata filters (Document Category, Year range slider from 1901 to 2026)
  - Top-K retrieved chunks slider
- [x] Collapsible Source Drawer beneath each response showing:
  - Retrieved excerpts
  - Similarity / RRF score
  - Case title, G.R. Number, or Republic Act Number
  - Summary / Ponente / Author details
- [x] Persistent Philippine Legal Disclaimer banner.

### Stage 7: Evaluation, Verification & Benchmarking (COMPLETED)
- [x] Exact keyword/statute queries benchmarked (e.g., Republic Act No. 10 penalties and provisions).
- [x] Conceptual/Jurisprudence queries benchmarked (e.g., Supreme Court rulings on demurrers).
- [x] Anti-hallucination guardrail benchmarked on out-of-domain queries (properly refusing with insufficient information notice).

---

## ⚖️ Legal AI Guardrails & Best Practices
1. **Source Grounding:** Responses must quote or link directly to the retrieved chunks.
2. **Deterministic Metadata Filtering:** Users can restrict queries to specific decades, courts, or statute types.
3. **Mandatory Disclaimer:** Visible warning that the tool is an AI research aid and not a substitute for licensed legal counsel.
