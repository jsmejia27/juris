# rag_pipeline.py
import os
import re
import logging
import threading
from typing import List, Dict, Any, Optional, Generator

from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import SparseTextEmbedding
from flashrank import Ranker, RerankRequest
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate

from open_congress import OpenCongressClient
from qdrant_service import ensure_qdrant_running, DEFAULT_QDRANT_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_STORAGE_DIR = "./qdrant_server_data"
DEFAULT_COLLECTION = "philippine_law"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "qwen3-embedding:4b"
DEFAULT_LLM_MODEL = "qwen3.5:9b"

SYSTEM_PROMPT_TEMPLATE = """You are **Juris**, an AI Legal Research Assistant specializing in **Philippine laws, statutes, Republic Acts, regulations, and Supreme Court jurisprudence**.

Your purpose is to explain Philippine legal information clearly, accurately, and practically using **ONLY the legal materials provided in the retrieved context**.

You are a retrieval-grounded legal assistant. The retrieved documents are your only authoritative source for the answer.{history_section}

==================================================
RETRIEVED LEGAL CONTEXT
=======================

{context}

==================================================
USER QUESTION
=============

{question}

==================================================
CORE RULE — STRICT LEGAL GROUNDING
==================================

You MUST base your answer only on the RETRIEVED LEGAL CONTEXT.

Do NOT use your general knowledge to fill gaps in the retrieved documents.

Do NOT invent or assume:

* laws
* Republic Acts
* Presidential Decrees
* Executive Orders
* Department Orders
* regulations
* Supreme Court decisions
* G.R. numbers
* case facts
* legal doctrines
* statutory sections
* penalties
* qualifications
* deadlines
* monetary amounts
* procedural requirements
* government agencies
* legal interpretations
* quotations
* URLs or links

If the answer cannot be established from the retrieved documents, clearly say:

"Based on the provided Philippine legal documents, there is insufficient information to answer this inquiry."

If only part of the question can be answered, answer the supported portion and clearly identify what information is missing.

Never manufacture an answer simply to appear helpful.

==================================================
ANSWER STYLE — STRUCTURED PHILIPPINE LEGAL EDITORIAL FORMAT
==================================================

Write the answer following a structured, comprehensive, and highly accessible Philippine legal editorial style.

The response should feel:

* authoritative, structured, and legally sound
* clear and practical for non-lawyers, employers, and employees
* organized into logical sections with clear bold headers, statutory callouts, and bulleted compliance requisites

Structure the answer as follows:

1. **Direct Overview & Legal Basis**: State the primary governing Philippine law directly in the opening paragraph. Bold all official statute names (e.g. **Solo Parents' Welfare Act of 2000 (Republic Act No. 8972)**).
2. **Numbered Provisions with Inline Citations**: Use descriptive numbered headings with inline citation tags (e.g. `#### 1. The Solo Parent Leave Benefit ( [RA 8972]¹ , as amended by [RA 11861]² )`). Explain the rule in short, readable paragraphs.
3. **Verbatim Statutory Excerpt**: Quote the exact section or rule from the retrieved documents inside a blockquote `> "..."` with proper attribution: `> — RA XXXX Section Y`.
4. **Key Details / Requisites for Compliance**: Summarize essential requisites or conditions using bullet points with bold lead-ins:
   * **Fully paid / Rate:** Specific compensation details.
   * **Eligibility / Service requirement:** Length of service or qualifications.
   * **Documentation required:** Specific IDs or certificates.
   * **Exceptions / Non-deductibility:** Relationship with other leaves or rules.
5. **Practical Takeaways / What You Should Do**: Practical guidance strictly grounded in the retrieved legal text.
6. **Case Law / Jurisprudence (when analyzing SC cases)**: Use standard Philippine case digest format: **Facts → Issue → Supreme Court Ruling → Legal Doctrine → Practical Meaning**.

==================================================
LANGUAGE & MULTILINGUAL SUPPORT (TAGALOG / FILIPINO / ENGLISH)
==================================================

1. If the user asks in Tagalog or Filipino, or explicitly requests the explanation in Tagalog/Filipino (e.g., "Ipaliwanag sa Tagalog", "Sagutin sa Filipino", "Ano ang mga karapatan ng...", "Ano ang batas tungkol sa..."):
   - **RESPOND DIRECTLY IN NATURAL, PROFESSIONAL FILIPINO / TAGALOG** (or natural Philippine Taglish for technical legal terminology).
   - Translate section headings, explanations, compliance requirements, and practical advice into clear, easy-to-understand Tagalog/Filipino.
   - Retain official legal statute titles, article numbers, and jurisprudence citations in their official form (e.g., **Expanded Solo Parents Welfare Act (Republic Act No. 11861)**, *Artikulo 36 ng Family Code*, *Psychological Incapacity*, *Probable Cause*, *G.R. No. L-XXXXX*).
   - For statutory blockquotes, quote the official text and provide a direct Tagalog explanation of what the provision means.
2. If the user asks in English, respond in English by default.

==================================================
OPENING
=======

Begin with a direct answer to the user's question.

The first paragraph should normally identify the principal law, regulation, or jurisprudence governing the issue.

Example style:

"The primary law governing single parents in the Philippines is the **Solo Parents' Welfare Act of 2000 (Republic Act No. 8972)**, as expanded by the **Expanded Solo Parents Welfare Act (Republic Act No. 11861)**."

Do not unnecessarily begin with labels such as:

"Executive Overview:"
"Legal Analysis:"
"Answer:"

unless such a heading genuinely improves the response.

==================================================
LEGAL REFERENCES
================

When mentioning a law, statute, Republic Act, or Supreme Court case, preserve the official name and citation exactly as supported by the retrieved documents.

Bold important legal authorities.

Examples:

**Solo Parents' Welfare Act of 2000 (Republic Act No. 8972)**

**Expanded Solo Parents Welfare Act (Republic Act No. 11861)**

**Article 95 of the Labor Code**

**G.R. No. XXXXX**

Never create or guess a citation.

==================================================
SECTION HEADINGS
================

Use Markdown headings when they improve readability.

Prefer descriptive headings that answer the user's question.

Examples:

#### 1. The Solo Parent Leave Benefit

#### 2. Who Qualifies as a Solo Parent?

#### 3. What Employers Need to Know

#### 4. When Benefits May Be Lost

#### 5. Practical Steps for Employers

Do not force a fixed number of sections.

Use as many sections as necessary, but avoid unnecessary repetition.

==================================================
NUMBERED LEGAL RULES
====================

When explaining multiple statutory rules, use numbered sections.

Each section should generally contain:

1. A short explanation of the rule.
2. The relevant legal authority.
3. The practical meaning of the rule.

Keep paragraphs relatively short.

Avoid large walls of text.

==================================================
INLINE CITATIONS
================

Use the citation information available in the retrieved context.

If the retrieval system provides citation identifiers, preserve them.

Preferred citation style:

[RA 8972]

[RA 11861]

[G.R. No. XXXXX]

If a citation marker is supplied by the retrieval system, use it exactly as provided.

Do not invent citation markers.

Attach citations to the legal proposition they support.

Example:

A qualified solo parent employee is entitled to the applicable parental leave benefit under **Republic Act No. 8972**, as amended by **Republic Act No. 11861**. [RA 8972] [RA 11861]

Do not place unrelated citations at the end of the entire answer.

==================================================
DIRECT QUOTATIONS
=================

Use a direct quotation ONLY when the retrieved legal context contains the relevant text.

Never reconstruct or fabricate a quotation from memory.

When a quotation is useful, format it as:

> "Exact text from the retrieved legal document."

> — **Section X, [Official Legal Authority]**

Keep quotations reasonably short and directly relevant.

If the retrieved context does not contain the exact wording of a provision, DO NOT present paraphrased text as a quotation.

Instead, explain the rule in your own words.

Do not use:

> svg

or any placeholder representing an image.

==================================================
PRACTICAL EXPLANATION
=====================

After explaining the legal rule, explain what it means in practice.

For example:

**Practical takeaway:** If an employee satisfies the legal requirements established in the retrieved documents, the employer must observe the applicable benefit. The employer should verify the requirements specified by the law or implementing rules.

Clearly distinguish between:

1. What the law expressly provides.
2. What is an interpretation of the provision.
3. What is practical guidance.

Never present an inference as an express statutory requirement.

==================================================
KEY DETAILS
===========

When useful, summarize important requirements using bullets.

Example:

**Key details:**

* **Eligibility:** Explain who qualifies based only on the retrieved documents.
* **Benefit:** Explain the benefit provided by the law.
* **Documentation:** Identify documents expressly required by the retrieved material.
* **Procedure:** Explain the procedure if provided.
* **Exceptions:** Identify applicable exceptions.
* **Duration:** State the applicable period if supported.
* **Termination:** Explain when eligibility ends if supported.
* **Penalties:** State penalties only if explicitly supported.

Do not create a requirement simply because it would be common administrative practice.

==================================================
PRACTICAL STEPS
===============

When the question involves an employer, employee, government agency, or compliance issue, provide practical steps when supported by the retrieved documents.

Example:

#### What an Employer Should Do

1. Verify the employee's eligibility.
2. Review the required documentation.
3. Apply the applicable leave or benefit.
4. Follow the procedure required by the law or implementing rules.
5. Maintain appropriate records.

However, ONLY include a step if it is supported by the retrieved legal context.

Do not introduce external HR practices as if they were legal requirements.

==================================================
CASE LAW
========

When answering questions involving Supreme Court jurisprudence, prioritize:

**Facts → Issue → Ruling → Doctrine → Practical Meaning**

Use this structure when applicable.

Example:

#### What Happened?

Briefly summarize the relevant facts from the retrieved case.

#### What Was the Issue?

State the legal issue addressed by the Court.

#### What Did the Supreme Court Rule?

Explain the Court's ruling.

#### Doctrine

State the legal doctrine established or applied by the Court, using only the retrieved text.

#### Practical Meaning

Explain what the ruling means in understandable terms.

Never invent facts or holdings not contained in the retrieved case materials.

==================================================
COMPARING LAWS
==============

When the user asks to compare laws, present the differences clearly.

Use a table when appropriate:

| Issue        | Law A | Law B |
| ------------ | ----- | ----- |
| Coverage     | ...   | ...   |
| Eligibility  | ...   | ...   |
| Benefit      | ...   | ...   |
| Requirements | ...   | ...   |

Every substantive entry must be supported by the retrieved documents.

If information for one law is missing, write:

"Not established in the provided documents."

Do not fill the gap from general knowledge.

==================================================
UNCERTAINTY AND CONFLICTS
=========================

If the retrieved documents contain conflicting information:

1. Identify the conflict.
2. Identify the provisions or cases involved.
3. Do not silently choose one version.
4. Explain that the retrieved materials appear inconsistent.

If a Supreme Court decision interprets or qualifies a statutory provision, explain that relationship when it is supported by the retrieved documents.

==================================================
IMPORTANT LEGAL DISTINCTIONS
============================

Do not automatically treat the following as equivalent:

* statutory right vs. company policy
* law vs. implementing rules
* eligibility vs. documentation
* legal requirement vs. recommended practice
* Supreme Court holding vs. commentary
* factual allegation vs. established fact
* general rule vs. exception

Clearly identify which one is being discussed.

==================================================
NO OVERCLAIMING
===============

Never say:

"Under Philippine law..."

unless the retrieved documents actually establish the proposition.

Prefer:

"Under **Republic Act No. ___**, as provided in the retrieved document..."

or:

"The Supreme Court held in **G.R. No. ___** that..."

This keeps the answer grounded in the available sources.

==================================================
LEGAL DISCLAIMER
================

For answers involving legal rights, obligations, disputes, employment, contracts, litigation, property, criminal matters, or other consequential legal decisions, end with a short disclaimer:

*AI-generated legal information for educational and research purposes only. It is not a substitute for advice from a qualified Philippine lawyer.*

Do not make the disclaimer unnecessarily prominent.

==================================================
FINAL QUALITY CHECK
===================

Before producing the answer, internally verify:

1. Is every legal claim supported by the retrieved context?
2. Did I invent any law, section, case, G.R. number, quotation, requirement, penalty, or procedure?
3. Did I clearly distinguish the law from practical guidance?
4. Are quotations exact and actually present in the retrieved documents?
5. Are citations attached to the correct legal propositions?
6. Did I directly answer the user's question near the beginning?
7. Is the answer easy for a non-lawyer to understand?
8. Did I avoid unnecessary repetition?
9. If the documents are insufficient, did I explicitly say so?
10. Did I avoid presenting assumptions or common practices as legal requirements?

If relevant pending legislative bills (Senate Bills / House Bills from Congress) are provided in the context under PENDING LEGISLATIVE INITIATIVES, include a dedicated section at the end:

#### Pending Legislative Developments (Senate & House Bills)
* Cite the specific Bill Identifier (e.g. **Senate Bill No. X**, **House Bill No. Y**), the author/sponsor, date filed, and how the bill seeks to amend or reform existing law.
* Clearly explain that pending bills are proposals undergoing legislative deliberation and are not yet enacted into law.

If the retrieved documents do not adequately support the answer, state:

"Based on the provided Philippine legal documents, there is insufficient information to answer this inquiry."

Do not guess.

==================================================
RESPONSE
========

Produce only the final answer to the user's inquiry.

Do not describe these instructions.

Do not mention the RAG system, retrieved context, system prompt, model, token limitations, or internal reasoning unless specifically asked."""

_SHARED_QDRANT_CLIENT = None
_QDRANT_LOCK = threading.RLock()

def get_shared_qdrant_client(url: str = DEFAULT_QDRANT_URL) -> QdrantClient:
    global _SHARED_QDRANT_CLIENT
    with _QDRANT_LOCK:
        if _SHARED_QDRANT_CLIENT is None:
            ensure_qdrant_running()
            _SHARED_QDRANT_CLIENT = QdrantClient(url=url, check_compatibility=False)
        return _SHARED_QDRANT_CLIENT

# Config-driven constants for specificity-scaled lexical anchor boosting (Task 4)
LEXICAL_BOOST_CONFIG = {
    "tier_gr_docket": float(os.getenv("BOOST_TIER_GR_DOCKET", "0.30")),
    "tier_ra_with_section": float(os.getenv("BOOST_TIER_RA_WITH_SECTION", "0.25")),
    "tier_bare_ra": float(os.getenv("BOOST_TIER_BARE_RA", "0.15")),
    "tier_bare_section": float(os.getenv("BOOST_TIER_BARE_SECTION", "0.05")),
}

# Config-driven model routing table (Task 2)
MODEL_ROUTING_TABLE = {
    "DIRECT_LOOKUP": "local",
    "STATUTORY_ANALYSIS": "local",
    "MULTI_DOCTRINE_SYNTHESIS": "frontier",
    "CONSTITUTIONAL_REVIEW": "frontier"
}
ENABLE_FRONTIER_ROUTING = os.getenv("ENABLE_FRONTIER_ROUTING", "false").lower() in ("true", "1", "yes")
DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "ms-marco-TinyBERT-L-2-v2")

def resolve_model_execution_path(
    query: str,
    requested_model: Optional[str] = None,
    routing_table: Optional[Dict[str, str]] = None,
    enable_frontier: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Routes query based on semantic complexity classifier in router.py.
    """
    from router import route_query
    q_route = route_query(query)
    complexity = q_route.get("complexity", "DIRECT_LOOKUP")

    routes = routing_table or MODEL_ROUTING_TABLE
    frontier_active = enable_frontier if enable_frontier is not None else ENABLE_FRONTIER_ROUTING
    target_tier = routes.get(complexity, "local")

    if target_tier == "frontier" and frontier_active:
        resolved_path = "frontier"
        resolved_model = os.getenv("FRONTIER_MODEL_NAME", "gemini-2.5-pro")
    else:
        resolved_path = "local"
        resolved_model = requested_model or DEFAULT_LLM_MODEL

    return {
        "execution_path": resolved_path,
        "model_name": resolved_model,
        "complexity": complexity,
        "route_reason": q_route.get("reason", ""),
        "frontier_enabled": frontier_active
    }

def extract_lexical_anchors_with_tiers(query: str) -> List[Dict[str, Any]]:
    """
    Extracts high-precision legal lexical anchors categorized into specificity tiers:
    - tier_gr_docket (+0.30): Full G.R. docket number (e.g. 'G.R. No. 196359')
    - tier_ra_with_section (+0.25): RA number + Section (e.g. 'RA 9262 Section 5(i)')
    - tier_bare_ra (+0.15): Bare RA number (e.g. 'RA 9262')
    - tier_bare_section (+0.05): Bare section/article (e.g. 'Section 5')
    """
    anchors = []
    if not query:
        return anchors

    # 1. G.R. Docket Numbers (Tier 1 - Highest Specificity)
    gr_matches = list(re.finditer(r'\b(?:G\.R\.|GR)(?:\s+No\.?)?\s*([L0-9\-]+)\b', query, re.IGNORECASE))
    for m in gr_matches:
        raw_num = m.group(1).strip()
        clean_num = raw_num.replace('-', '')
        anchors.append({
            "tier": "tier_gr_docket",
            "terms": [f"g.r. {raw_num.lower()}", f"gr {raw_num.lower()}", raw_num.lower(), clean_num.lower()],
            "raw": m.group(0)
        })

    # 2. RA with Section (Tier 2 - High Specificity)
    ra_sec_matches = []
    # Pattern A: RA 9262 Section 5
    for m in re.finditer(r'\b(?:Republic\s+Act(?:\s+No\.?)?|RA|R\.A\.?)\s*(\d{3,6})\s+(?:Section|Sec\.|Article|Art\.)\s*([0-9A-Za-z\-\(\)]+)', query, re.IGNORECASE):
        ra_num = m.group(1).strip()
        sec_val = m.group(2).strip().lower()
        ra_sec_matches.append({
            "tier": "tier_ra_with_section",
            "ra_num": ra_num,
            "sec_val": sec_val,
            "terms": [f"ra {ra_num}", f"section {sec_val}", f"sec. {sec_val}", f"article {sec_val}"],
            "raw": m.group(0)
        })
    # Pattern B: Section 5 of RA 9262
    for m in re.finditer(r'\b(?:Section|Sec\.|Article|Art\.)\s*([0-9A-Za-z\-\(\)]+)\s+(?:of|under|in)\s+(?:Republic\s+Act(?:\s+No\.?)?|RA|R\.A\.?)\s*(\d{3,6})', query, re.IGNORECASE):
        sec_val = m.group(1).strip().lower()
        ra_num = m.group(2).strip()
        ra_sec_matches.append({
            "tier": "tier_ra_with_section",
            "ra_num": ra_num,
            "sec_val": sec_val,
            "terms": [f"ra {ra_num}", f"section {sec_val}", f"sec. {sec_val}", f"article {sec_val}"],
            "raw": m.group(0)
        })
    anchors.extend(ra_sec_matches)

    # 3. Bare RA (Tier 3 - Medium Specificity)
    ra_matches = list(re.finditer(r'\b(?:Republic\s+Act(?:\s+No\.?)?|RA|R\.A\.?)\s*(\d{3,6})\b', query, re.IGNORECASE))
    for m in ra_matches:
        ra_num = m.group(1).strip()
        anchors.append({
            "tier": "tier_bare_ra",
            "terms": [f"ra {ra_num}", f"republic act {ra_num}", ra_num],
            "raw": m.group(0)
        })

    # 4. Bare Section / Article (Tier 4 - Low Specificity)
    sec_matches = list(re.finditer(r'\b(?:Section|Sec\.|Article|Art\.)\s*([0-9A-Za-z\-\(\)]+)', query, re.IGNORECASE))
    for m in sec_matches:
        sec_val = m.group(1).strip().lower()
        prefix = "article" if "art" in m.group(0).lower() else "section"
        anchors.append({
            "tier": "tier_bare_section",
            "terms": [f"{prefix} {sec_val}", f"section {sec_val}.", f"section {sec_val} ", f"article {sec_val}.", f"article {sec_val} "],
            "raw": m.group(0)
        })

    return anchors

def extract_lexical_anchors(query: str) -> List[str]:
    """Backward-compatible flat extractor returning string list."""
    anchors_with_tiers = extract_lexical_anchors_with_tiers(query)
    out = []
    for a in anchors_with_tiers:
        out.extend(a.get("terms", []))
    return list(dict.fromkeys(out))

def apply_lexical_anchor_boost(
    candidates: List[Dict[str, Any]],
    query: str,
    config: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Tiered Lexical Anchor Boost:
    Calculates specific boost based on anchor hierarchy (G.R. > RA+Section > Bare RA > Bare Section).
    Logs the firing tier and matched tokens onto the candidate document object.
    """
    boost_cfg = config or LEXICAL_BOOST_CONFIG
    tiered_anchors = extract_lexical_anchors_with_tiers(query)
    if not tiered_anchors:
        return candidates

    scored_candidates = []
    for doc in candidates:
        text_corpus = (
            str(doc.get("title", "")) + " " +
            str(doc.get("gr_no", "")) + " " +
            str(doc.get("doc_id", "")) + " " +
            str(doc.get("text", ""))
        ).lower()

        highest_boost = 0.0
        fired_tier = None
        matched_anchors = []

        for anc in tiered_anchors:
            tier_name = anc["tier"]
            boost_val = boost_cfg.get(tier_name, 0.0)

            if tier_name == "tier_ra_with_section":
                ra_found = f"ra {anc['ra_num']}" in text_corpus or f"republic act {anc['ra_num']}" in text_corpus or anc["ra_num"] in text_corpus
                sec_found = f"section {anc['sec_val']}" in text_corpus or f"sec. {anc['sec_val']}" in text_corpus or f"article {anc['sec_val']}" in text_corpus or f"section {anc['sec_val']}." in text_corpus or f"section {anc['sec_val']} " in text_corpus
                if ra_found and sec_found:
                    if boost_val > highest_boost:
                        highest_boost = boost_val
                        fired_tier = "tier_ra_with_section"
                    matched_anchors.append(anc["raw"])
            else:
                matched = any(t in text_corpus for t in anc["terms"])
                if matched:
                    if boost_val > highest_boost:
                        highest_boost = boost_val
                        fired_tier = tier_name
                    matched_anchors.append(anc["raw"])

        doc_copy = dict(doc)
        new_score = float(doc.get("score", 0.0)) + highest_boost
        doc_copy["score"] = round(new_score, 4)
        doc_copy["lexical_boost"] = highest_boost
        doc_copy["boost_tier"] = fired_tier
        doc_copy["matched_anchors"] = matched_anchors
        scored_candidates.append(doc_copy)

    # Sort descending by boosted score
    scored_candidates.sort(key=lambda d: d["score"], reverse=True)
    return scored_candidates

class LegalCrossEncoderRanker:
    """
    Unified Cross-Encoder Reranker supporting:
    1. 'BAAI/bge-reranker-base' via FastEmbed (~500MB ONNX, high precision)
    2. 'ms-marco-TinyBERT-L-2-v2' via FlashRank (lightweight baseline)
    """
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        self._bge_ranker = None
        self._flash_ranker = None

        if "bge" in model_name.lower():
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                self.engine_type = "bge"
                self._bge_ranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base")
                logger.info("Initialized BGE-Reranker-Base via FastEmbed ONNX.")
            except Exception as e:
                logger.warning(f"FastEmbed BGE reranker init failed ({e}), falling back to FlashRank.")
                self.engine_type = "flashrank"
                self._flash_ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
        else:
            self.engine_type = "flashrank"
            self._flash_ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
            logger.info("Initialized FlashRank (ms-marco-TinyBERT-L-2-v2).")

    def rerank_passages(self, query: str, candidate_docs: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        if not candidate_docs:
            return []

        doc_passages = [
            f"[{doc.get('category', '')}] {doc.get('title', '')} ({doc.get('gr_no', '')}): {doc.get('text', '')}"
            for doc in candidate_docs
        ]

        if self.engine_type == "bge" and self._bge_ranker is not None:
            results = list(self._bge_ranker.rerank(query, doc_passages))
            indexed_scores = [(idx, float(score)) for idx, score in enumerate(results)]
            sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)[:top_k]
            reranked_docs = []
            for idx, score in sorted_scores:
                doc = dict(candidate_docs[idx])
                doc["rerank_score"] = score
                doc["score"] = score
                doc["reranker_model"] = "bge-reranker-base"
                reranked_docs.append(doc)
            return reranked_docs
        else:
            from flashrank import RerankRequest
            passages = [{"id": i, "text": p} for i, p in enumerate(doc_passages)]
            req = RerankRequest(query=query, passages=passages)
            results = self._flash_ranker.rerank(req)
            reranked_docs = []
            for item in results[:top_k]:
                idx = item["id"]
                doc = dict(candidate_docs[idx])
                doc["rerank_score"] = float(item["score"])
                doc["score"] = float(item["score"])
                doc["reranker_model"] = "ms-marco-TinyBERT-L-2-v2"
                reranked_docs.append(doc)
            return reranked_docs

def clean_citation_title(title: str, gr_no: str = "", category: str = "") -> str:
    """
    Cleans raw docket or database filenames (e.g. 'gr_203335_so_2014') into proper judicial citation titles.
    """
    if not title or not title.strip():
        return gr_no or "Philippine Legal Authority"

    t = title.strip()
    if t.lower().startswith("gr_") or t.lower().startswith("ra_") or t.lower().endswith(".html") or t.lower().endswith(".php"):
        m = re.search(r'(?:gr|ra)_(\d+)', t.lower())
        if m:
            prefix = "Republic Act No." if "ra" in t.lower() else "G.R. No."
            cat = "Republic Act" if "ra" in t.lower() else "Supreme Court Decision"
            return f"{prefix} {m.group(1)} ({cat})"
    return t

def deduplicate_sources(sources_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicates legal sources and citations by canonical G.R. number, law number, or title.
    Retains the first/highest-scoring entry for each unique legal authority.
    """
    if not sources_list:
        return []

    seen = set()
    deduped = []

    for s in sources_list:
        gr_raw = str(s.get("gr_no") or "").strip().lower()
        gr_clean = re.sub(r'[^a-z0-9]', '', gr_raw)

        law_raw = str(s.get("law_no") or "").strip().lower()
        law_clean = re.sub(r'[^a-z0-9]', '', law_raw)

        doc_id = str(s.get("doc_id") or "").strip().lower()
        title_raw = str(s.get("title") or "").strip().lower()

        key = gr_clean or law_clean or doc_id or title_raw
        if key and key in seen:
            continue
        if key:
            seen.add(key)

        s_copy = dict(s)
        s_copy["title"] = clean_citation_title(s_copy.get("title", ""), s_copy.get("gr_no", ""), s_copy.get("category", ""))
        deduped.append(s_copy)

    return deduped

class LegalRetriever:
    def __init__(
        self,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        collection_name: str = DEFAULT_COLLECTION,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        client: Optional[QdrantClient] = None
    ):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.client = client or get_shared_qdrant_client(qdrant_url)

        # Check collection vector dimension if collection exists
        actual_embed_model = embed_model
        try:
            if self.client.collection_exists(collection_name):
                col_info = self.client.get_collection(collection_name)
                vectors_cfg = col_info.config.params.vectors
                col_dim = None
                if hasattr(vectors_cfg, "dense"):
                    col_dim = vectors_cfg.dense.size
                elif isinstance(vectors_cfg, dict) and "dense" in vectors_cfg:
                    col_dim = vectors_cfg["dense"].size if hasattr(vectors_cfg["dense"], "size") else vectors_cfg["dense"].get("size")
                elif hasattr(vectors_cfg, "size"):
                    col_dim = vectors_cfg.size

                if col_dim == 768 and "qwen3-embedding" in embed_model:
                    logger.warning(
                        f"Collection '{collection_name}' has 768-dim vectors (indexed with nomic-embed-text). "
                        "Using nomic-embed-text for query retrieval compatibility."
                    )
                    actual_embed_model = "nomic-embed-text:latest"
                elif col_dim == 2560 and "nomic" in embed_model:
                    actual_embed_model = "qwen3-embedding:4b"
        except Exception as e:
            logger.debug(f"Could not inspect collection vector dimension: {e}")

        self.dense_embedder = OllamaEmbeddings(model=actual_embed_model, base_url=ollama_url)
        self.sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
        self.ranker = LegalCrossEncoderRanker(model_name=reranker_model)

    def retrieve(
        self,
        query: str,
        limit: int = 8,
        category: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        ponente: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Two-Stage High-Precision Retrieval:
        1. Hybrid Search (Dense + Sparse BM25 via RRF) to fetch top 50 candidate passages.
        2. Tiered Lexical Anchor Boost (specificity-scaled boost for G.R. / RA / Section).
        3. Cross-Encoder Re-Ranker (BGE-Reranker-Base / FlashRank) to score exact legal relevance.
        4. Doctrine Currency Filter (deprioritizing reversed/abandoned cases unless historical).
        """
        must_conditions = []
        if category and category.lower() != "all":
            must_conditions.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category)
                )
            )
        if year_min is not None or year_max is not None:
            range_cond = {}
            if year_min is not None:
                range_cond["gte"] = year_min
            if year_max is not None:
                range_cond["lte"] = year_max
            must_conditions.append(
                models.FieldCondition(
                    key="year",
                    range=models.Range(**range_cond)
                )
            )
        if ponente and ponente.strip():
            must_conditions.append(
                models.FieldCondition(
                    key="ponente",
                    match=models.MatchText(text=ponente.strip())
                )
            )

        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        # 1. Stage 1: Candidate Search (Dense + Sparse RRF over top 50)
        q_dense = self.dense_embedder.embed_query(query)
        q_sparse = list(self.sparse_embedder.embed([query]))[0]
        candidate_limit = 50

        with _QDRANT_LOCK:
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=q_dense,
                        using="dense",
                        limit=candidate_limit,
                        filter=query_filter
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=q_sparse.indices.tolist(),
                            values=q_sparse.values.tolist()
                        ),
                        using="sparse",
                        limit=candidate_limit,
                        filter=query_filter
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=candidate_limit
            )

        candidate_docs = []
        for point in results.points:
            payload = point.payload or {}
            candidate_docs.append({
                "score": point.score,
                "doc_id": payload.get("doc_id", ""),
                "title": payload.get("title", ""),
                "gr_no": payload.get("gr_no", ""),
                "category": payload.get("category", ""),
                "date": payload.get("date", ""),
                "year": payload.get("year", ""),
                "ponente": payload.get("ponente", ""),
                "summary": payload.get("summary", ""),
                "key_provisions": payload.get("key_provisions", ""),
                "chunk_index": payload.get("chunk_index", 1),
                "total_chunks": payload.get("total_chunks", 1),
                "text": payload.get("text", "")
            })

        if not candidate_docs:
            return []

        # 2. Stage 2: Specificity-Scaled Tiered Lexical Anchor Boosting
        boosted_candidates = apply_lexical_anchor_boost(candidate_docs, query)

        # 3. Stage 3: Neural Cross-Encoder Re-Ranking (Top-50 candidates -> Top-8)
        try:
            reranked_docs = self.ranker.rerank_passages(query, boosted_candidates[:50], top_k=limit*2)
        except Exception as e:
            logger.warning(f"Re-ranking exception, falling back to boosted candidates: {e}")
            reranked_docs = boosted_candidates

        # 4. Stage 4: Doctrine Currency Filtering & Tagging
        from doctrine_currency import filter_and_tag_doctrine_currency
        currency_filtered = filter_and_tag_doctrine_currency(reranked_docs, query)

        return currency_filtered[:limit]

DEFAULT_NUM_CTX = 16384  # Expanded 16K context budget for RTX 5070 Ti (16GB VRAM)
DEFAULT_TEMPERATURE = 0.0

class LegalRAGPipeline:
    def __init__(
        self,
        retriever: Optional[LegalRetriever] = None,
        llm_model: str = DEFAULT_LLM_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        num_ctx: int = DEFAULT_NUM_CTX,
        congress_client: Optional[OpenCongressClient] = None
    ):
        self.retriever = retriever or LegalRetriever(ollama_url=ollama_url)
        self.congress_client = congress_client or OpenCongressClient()
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._init_llm()

    def _init_llm(self):
        self.llm = OllamaLLM(
            model=self.llm_model,
            base_url=self.ollama_url,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            num_gpu=99  # Full GPU offloading on RTX 5070 Ti
        )

    def set_model(self, model_name: str):
        if model_name != self.llm_model:
            self.llm_model = model_name
            self._init_llm()

    def set_temperature(self, temperature: float):
        if temperature != self.temperature:
            self.temperature = temperature
            self._init_llm()

    def set_num_ctx(self, num_ctx: int):
        if num_ctx != self.num_ctx:
            self.num_ctx = num_ctx
            self._init_llm()

    @staticmethod
    def extract_section(text: str) -> str:
        match = re.search(r'\b(?:Section|Sec\.|Article|Art\.)\s*([0-9A-Za-z\-\(\)]+)', text, re.IGNORECASE)
        if match:
            prefix = "Article" if "art" in match.group(0).lower() else "Section"
            return f"{prefix} {match.group(1)}"
        return "General Provisions"

    @staticmethod
    def format_history_section(history: Optional[List[Dict[str, str]]]) -> str:
        """
        Formats previous conversation turns into structured text for multi-turn prompt awareness.
        """
        if not history:
            return ""
        formatted = []
        for turn in history[-6:]: # Keep up to last 6 turns
            role = "User" if turn.get("role") == "user" else "Juris"
            content = (turn.get("content") or "").strip()
            if content:
                # Truncate overly long assistant turns in history to preserve context budget
                if len(content) > 600 and role == "Juris":
                    content = content[:600] + "..."
                formatted.append(f"**{role}**: {content}")
        if not formatted:
            return ""
        return "\n\n==================================================\nPRIOR CONVERSATION HISTORY (RECENT TURNS)\n==================================================\n\n" + "\n\n".join(formatted) + "\n"

    @staticmethod
    def contextualize_query(query: str, history: Optional[List[Dict[str, str]]]) -> str:
        """
        If the current query is a short follow-up (contains pronouns or relative terms),
        enrich the search query with subject keywords from the immediate previous user query.
        """
        if not history or not query:
            return query

        follow_up_cues = [
            "that", "this", "it", "they", "those", "these", "penalties", "penalty", "punishment",
            "exemptions", "exceptions", "benefits", "requirements", "requisites", "procedure",
            "how about", "what about", "who qualifies", "amended", "effectivity", "rules", "grounds",
            "yun", "iyon", "iyan", "nito", "niyan", "nila", "ano naman", "paano kung", "kailan ito"
        ]
        q_lower = query.lower()
        words = query.strip().split()
        is_follow_up = len(words) <= 6 or any(cue in q_lower for cue in follow_up_cues)

        if is_follow_up:
            past_user_queries = [t.get("content", "").strip() for t in history if t.get("role") == "user" and t.get("content", "").strip()]
            if past_user_queries:
                last_q = past_user_queries[-1]
                # Return combined query for hybrid retrieval
                return f"{last_q} {query}"

        return query

    def format_context(self, docs: List[Dict[str, Any]], bills_context: str = "", max_chars: int = 40000) -> str:
        if not docs and not bills_context:
            return "No relevant legal documents found."
        
        formatted_sources = []
        current_chars = 0

        for i, doc in enumerate(docs, 1):
            doc_type = doc.get("category") or "Republic Act"
            title = doc.get("title") or "Philippine Legal Document"
            citation = doc.get("gr_no") or "N/A"
            section = doc.get("section") or self.extract_section(doc.get("text", ""))
            date = doc.get("date") or str(doc.get("year") or "")
            doc_status = (doc.get("doctrine_status") or "good_law").upper()
            source_url = doc.get("source_url") or doc.get("doc_id") or ""
            if source_url and not source_url.startswith("http"):
                clean_path = source_url.replace("juris:", "").replace("repacts:", "")
                source_url = f"https://lawphil.net/{clean_path}"

            body = doc.get("text", "").strip()
            if doc.get("summary"):
                body = f"[Official Summary: {doc['summary']}]\n\n{body}"

            source_block = f"""SOURCE {i}
Type: {doc_type}
Title: {title}
Citation: {citation}
Section: {section}
Date: {date}
Doctrine Status: {doc_status}
Source URL: {source_url}
Text:
{body}"""
            if current_chars + len(source_block) > max_chars:
                # Clip last block to fit budget
                remaining = max_chars - current_chars
                if remaining > 200:
                    formatted_sources.append(source_block[:remaining] + "\n... [Remaining text clipped to 10K token budget]")
                break

            formatted_sources.append(source_block)
            current_chars += len(source_block)

        base_context = "\n\n" + ("\n" + "=" * 50 + "\n\n").join(formatted_sources)
        if bills_context:
            base_context += f"\n\n{'=' * 50}\n\n{bills_context}"

        return base_context

    def query(
        self,
        question: str,
        limit: int = 5,
        category: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        ponente: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes full retrieval and non-streaming generation.
        """
        docs = self.retriever.retrieve(
            query=question,
            limit=limit,
            category=category,
            year_min=year_min,
            year_max=year_max,
            ponente=ponente
        )
        context_str = self.format_context(docs)
        prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_str, question=question)
        response_text = self.llm.invoke(prompt)

        return {
            "question": question,
            "answer": response_text,
            "sources": docs,
            "prompt": prompt
        }

    def stream_query(
        self,
        question: str,
        limit: int = 5,
        category: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        ponente: Optional[str] = None
    ) -> Generator[str, None, List[Dict[str, Any]]]:
        """
        Executes retrieval and yields streaming token chunks from Ollama.
        """
        docs = self.retriever.retrieve(
            query=question,
            limit=limit,
            category=category,
            year_min=year_min,
            year_max=year_max,
            ponente=ponente
        )
        context_str = self.format_context(docs)
        prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_str, question=question)
        
        # Stream response
        for chunk in self.llm.stream(prompt):
            yield chunk

        return docs
