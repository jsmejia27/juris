# ⚖️ Juris — Sovereign Philippine Legal AI Platform & Research Workspace

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qdrant Vector DB](https://img.shields.io/badge/Qdrant-1.13.4-red.svg)](https://qdrant.tech/)
[![Ollama](https://img.shields.io/badge/LLM-Qwen%203.5%209B-green.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Juris** is an institutional, local-first Philippine Legal AI research platform and legal intelligence workspace. It combines dense semantic vector retrieval and BM25 sparse search across the complete body of Philippine primary law (**113,010 documents** / **2,360,407 vector points** indexed in native **Qdrant**) with live 20th Congress legislative tracking via the **BetterGov.ph Open Congress API** and sovereign local LLM synthesis (**Qwen 3.5 9B** on NVIDIA RTX).

Official Production Domain: [https://juris.bettermangaldan.org](https://juris.bettermangaldan.org)

---

## 🌟 Key Features

* **Complete 113,010 Philippine Legal Corpus:** 100% indexed statutory, jurisprudential, and executive database (1901–2026) with **2.36M vector chunks** stored locally in high-performance standalone Qdrant:
  * **Primary Statutory & Case Law (79,720):** Supreme Court Decisions (66,559), Republic Acts (12,065), Batas Pambansa (887), Commonwealth Acts (733), Historical Public Acts (4,257), 2025 RAs (200), Constitutions (9).
  * **Executive & Administrative Issuances (33,290):** Proclamations (13,195), Executive Orders (5,747), Administrative Orders (2,811), Memorandum Orders (2,401), Memorandum Circulars (2,220), Presidential Decrees (1,845), General Orders (81).
* **Dual-Stage Verification & Zero Fabrication:** 
  * Cross-encoder re-ranking via **FlashRank** (`ms-marco-TinyBERT-L-2-v2`).
  * Automated two-stage claim verification with Levenshtein quote matching to prevent hallucinations.
* **Hybrid Dual-Stream Retrieval:** Concurrent vector search across local statutory archives paired with real-time bill tracking from the **BetterGov.ph Open Congress API**.
* **Editorial Judicial Design System:** High-contrast editorial typography (*Playfair Display*, *Plus Jakarta Sans*, *JetBrains Mono*) with collapsible reasoning traces, ochre-gold statutory callout blocks, and source inspector modals.
* **Sovereign Persona ("Meet Juris"):** Full character showcase embodying Philippine legal intelligence with zero fabrication safeguards and bilingual fluency (English & Filipino/Tagalog).
* **Rate Limiting & Input Protection:** Immediate input locking during synthesis and a **60-second cooldown timer** between questions to prevent duplicate requests and spamming.
* **Premium PDF Memorandum Generator:** One-click export of research sessions or individual turns into official CPRA-compliant legal memorandums with header download attribution (`https://juris.bettermangaldan.org`) and full legal disclaimers.
* **Enterprise Security Hardening:**
  * Environment-driven admin authentication (`JURIS_ADMIN_USER` & `JURIS_ADMIN_PASSWORD`).
  * Qdrant bound strictly to `127.0.0.1` with static content hosting disabled.
  * Restricted CORS origin whitelisting.
  * Strict Pydantic V2 input validation (`num_ctx`, `top_k`, `temperature`, allowed models).
  * Client-side DOMPurify sanitization against XSS.
  * Automated size-based log file rotation.

---

## 🏛️ System Architecture

```
User Legal Query (Desktop / Mobile Workspace)
                    │
                    ▼
       Query Contextualizer & Router
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
Native Qdrant Engine     BetterGov.ph API
(113,010 Legal Docs)    (20th Congress Bills)
  - 79,720 Statutes/SC    - Live Senate Feeds
  - 33,290 Exec Orders    - Live House Feeds
        │                       │
        └───────────┬───────────┘
                    ▼
     FlashRank Cross-Encoder Re-Ranker
                    ▼
     Judicial Editorial Prompt Builder
                    ▼
         Local Ollama LLM Engine
        (Qwen 3.5 9B / RTX 5070 Ti)
                    ▼
       Two-Stage Claim Verification
        (Levenshtein Quote Matching)
                    ▼
    Async SSE Stream (/api/chat/stream)
                    ▼
  Interactive Workspace & PDF Memorandum
```

---

## 📊 Indexed Legal Corpus Breakdown

| Category | Document Count | Span | Type |
| :--- | :--- | :--- | :--- |
| **Supreme Court Decisions** | **66,559** | 1901 – 2024 | Primary Case Law & Jurisprudence |
| **Republic Acts** | **12,065** | 1946 – 2024 | Principal Statutory Law |
| **Proclamations** | **13,195** | 1935 – 2024 | Executive Issuance |
| **Executive Orders (EO)** | **5,747** | 1935 – 2024 | Executive Issuance |
| **Historical Public Acts** | **4,257** | 1900 – 1935 | Philippine Commission & Legislature |
| **Administrative Orders (AO)** | **2,811** | 1936 – 2024 | Executive & Agency Issuance |
| **Memorandum Orders (MO)** | **2,401** | 1958 – 2024 | Office of the President |
| **Memorandum Circulars (MC)**| **2,220** | 1960 – 2024 | Executive Directives |
| **Presidential Decrees (PD)** | **1,845** | 1972 – 1986 | Statutory & Martial Law Decrees |
| **Batas Pambansa (BP)** | **887** | 1978 – 1985 | Parliamentary Statutes |
| **Commonwealth Acts (CA)** | **733** | 1935 – 1946 | Commonwealth Era Statutes |
| **2025 Republic Acts** | **200** | 2025 | Modern Congress Statutes |
| **General Orders (GO)** | **81** | 1972 – 1985 | Armed Forces & Presidential Directives |
| **Philippine Constitutions** | **9** | 1899 – 1987 | Fundamental Organic Law |
| **TOTAL VERIFIED CORPUS** | **113,010** | **1901 – 2026** | **2,360,407 Vector Chunks** |

---

## 🚀 Quickstart

### Prerequisites
* **Python:** 3.10+
* **Ollama:** Running locally on port `11434` with models:
  ```bash
  ollama pull qwen3.5:9b
  ollama pull nomic-embed-text
  ```
* **Qdrant:** Native binary (`./bin/qdrant.exe`) or Docker running on port `6333`.

### Environment Configuration
Create a `.env` file or export environment variables:
```bash
JURIS_ADMIN_USER=admin
JURIS_ADMIN_PASSWORD=YourSecurePassword2026
JURIS_ALLOWED_ORIGINS=https://juris.bettermangaldan.org,http://localhost:9010
```

### Installation & Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/jsmejia27/juris.git
   cd juris
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv legal_ai_env
   # On Windows:
   .\legal_ai_env\Scripts\activate
   # On Linux/macOS:
   source legal_ai_env/bin/activate
   pip install -r requirements.txt
   ```

3. **Start the Qdrant Vector Engine:**
   ```bash
   .\bin\qdrant.exe --config-path .\config\config.yaml
   ```

4. **Launch the Juris Application Server:**
   ```bash
   python server.py
   ```

5. **Access Endpoints:**
   * **Main Workspace & Landing:** `http://localhost:9010/`
   * **Dedicated Research Chat:** `http://localhost:9010/chat`
   * **Legal Disclaimer & Terms:** `http://localhost:9010/disclaimer`
   * **Management Console:** `http://localhost:9010/manage-juris`

---

## 🔒 Security Hardening

Juris incorporates strict security protocols suitable for judicial and institutional deployment:
* **Admin Guard:** Secure HTTP Basic Auth protecting all dataset ingestion and server health APIs.
* **XSS Sanitization:** All AI-rendered markdown and HTML outputs are sanitized client-side via **DOMPurify**.
* **Payload Constraints:** Strict Pydantic Field validation limits user queries to $\le 4,000$ characters, restricts token context windows ($1,024–16,384$), and whitelists approved LLM identifiers.
* **Network Isolation:** Qdrant service is bound exclusively to `127.0.0.1`.

---

## ⚖️ Legal Disclaimer & CPRA Notice

Juris is an automated artificial intelligence research system designed strictly for educational, academic, and research reference. **Juris is not a law firm or a licensed member of the Integrated Bar of the Philippines (IBP)**. Use of this platform does not establish an attorney-client relationship under Canon II of the *Code of Professional Responsibility and Accountability (CPRA)*. Users must independently verify all statutory numbers, section references, and G.R. citations with primary publications in the *Official Gazette* or the *Supreme Court E-Library*.

---

## 🤝 Attribution & Acknowledgements

* **BetterGov.ph:** Data sourced from and indexed in collaboration with the [BetterGov.ph](https://bettergov.ph) Open Legal & Congress Initiative.
* **Supreme Court of the Philippines & Official Gazette:** Primary legal texts and decisions.
* **Qdrant & Ollama:** High-performance vector retrieval and local inference infrastructure.

---

## 📄 License
Released under the [MIT License](LICENSE). Synthesized for sovereign legal research, transparency, and civic technology in the Republic of the Philippines.
