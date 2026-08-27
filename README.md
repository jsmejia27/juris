# ⚖️ Juris — Philippine Legal AI Platform & Research Workspace

**Juris** is an open-source, local-first Philippine Legal AI research platform. It combines dense semantic retrieval against the entire corpus of Philippine law (**12,000+ Republic Acts** and **66,759 Supreme Court Decisions** indexed in native **Qdrant**) with live legislative tracking from the **20th Congress** and local LLM synthesis (**Qwen 3.5 9B** / **Qwen3-Embedding 4B**).

---

## 🌟 Key Features

* **Complete Philippine Legal Corpus:** 100% indexed statutory and jurisprudential database with over **2,821,455 vector points** stored locally in high-performance standalone Qdrant.
* **Hybrid Dual-Stream Retrieval:** Concurrent vector search across local statutory archives paired with real-time pending bill searches from the **BetterGov.ph Open Congress API**.
* **Editorial Judicial Design System:** High-contrast editorial typography (*Playfair Display*, *Plus Jakarta Sans*, *JetBrains Mono*) with collapsible reasoning traces, ochre-gold statutory callout blocks, and source inspector modals.
* **Multi-Turn Conversational Memory:** Dynamic context resolution for follow-up legal questions (*e.g., What are the penalties for that?*).
* **Premium PDF Memorandum Generator:** One-click export of the entire research session or individual query turns into official CPRA-compliant legal memorandums.
* **Secured Management Dashboard:** Protected admin operations (/manage-juris) for real-time vector collection health and statistics.

---

## 🏛️ System Architecture

`
User Inquiry ──► Query Contextualizer (Multi-Turn Buffer)
                        │
       ┌────────────────┴────────────────┐
       ▼                                 ▼
Local Qdrant Vector Engine       BetterGov Open Congress API
(12k+ RAs • 66k+ SC Decisions)  (20th Congress Senate/House Bills)
       │                                 │
       └────────────────┬────────────────┘
                        ▼
         Judicial Editorial Prompt Builder
                        ▼
             Local Ollama LLM Engine
          (qwen3.5:9b on RTX 5070 Ti)
                        ▼
         Async SSE Stream (/api/chat/stream)
                        ▼
   Desktop / Mobile UI + PDF Memorandum Export
`

---

## 🚀 Quickstart

### Prerequisites
* **Python:** 3.10+
* **Ollama:** Running locally on port 11434 with models:
  `ash
  ollama pull qwen3.5:9b
  ollama pull nomic-embed-text
  ollama pull qwen3-embedding:4b
  `
* **Qdrant:** Native binary or Docker instance running on port 6333.

### Setup & Run
1. **Clone the repository:**
   `ash
   git clone https://github.com/jsmejia27/juris.git
   cd juris
   `

2. **Install dependencies:**
   `ash
   python -m venv venv
   source venv/bin/activate  # Or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   `

3. **Start the application:**
   `ash
   python server.py
   `

4. **Access the application:**
   * **Homepage & Assistant:** http://localhost:9010
   * **Dedicated Legal Workspace:** http://localhost:9010/chat
   * **Management Console:** http://localhost:9010/manage-juris

---

## 📄 License
MIT License. Synthesized for research, educational, and legal tech experimentation in the Philippines.
