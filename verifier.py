# verifier.py - Structured Output Parser & Citation Verifier
import os
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

VERIFICATION_LOG_PATH = "logs/verification_failures.jsonl"

class Claim(BaseModel):
    text: str
    source_id: str = Field(description="Point ID, doc_id, law_no, or G.R. number of the retrieved legal authority")
    quoted_text: Optional[str] = Field(default=None, description="Exact verbatim excerpt if directly quoted, else null")
    source_type: Literal["statute", "jurisprudence", "bill", "unknown"] = "unknown"
    verified: Optional[bool] = None
    verification_reason: Optional[str] = None
    similarity_score: Optional[float] = None
    failure_type: Optional[Literal["quote_mismatch", "source_not_retrieved"]] = None

class StructuredLegalResponse(BaseModel):
    claims: List[Claim] = Field(default_factory=list)
    answer_prose: str = Field(description="Rendered natural-language Philippine legal editorial response")

class VerificationSummary(BaseModel):
    total_claims: int
    verified_claims: int
    unverified_claims: int
    accuracy_rate: float
    failures: List[Dict[str, Any]] = Field(default_factory=list)
    annotated_prose: str

def normalize_text_for_matching(text: str) -> str:
    if not text:
        return ""
    t = text.replace('“', '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = t.replace("—", " - ").replace("–", " - ")
    t = re.sub(r'\s+', ' ', t).strip().lower()
    return t

def parse_and_validate_structured_output(raw_output: str) -> StructuredLegalResponse:
    if not raw_output or not raw_output.strip():
        return StructuredLegalResponse(claims=[], answer_prose="No output generated.")

    clean_text = raw_output.strip()

    # 1. Try to extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', clean_text)
    if json_match:
        json_candidate = json_match.group(1).strip()
    else:
        brace_start = clean_text.find('{')
        brace_end = clean_text.rfind('}')
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            json_candidate = clean_text[brace_start:brace_end+1].strip()
        else:
            json_candidate = ""

    if json_candidate:
        try:
            parsed = json.loads(json_candidate)
            if isinstance(parsed, dict) and "answer_prose" in parsed:
                return StructuredLegalResponse(**parsed)
        except Exception as err:
            logger.debug(f"JSON parsing failed on candidate: {err}")

    # 2. Fallback heuristic
    logger.warning("Structured JSON parsing failed. Falling back to heuristic prose extraction.")
    claims = []
    cite_matches = re.finditer(r'\[([A-Za-z0-9\s\.\,\-\–]+)\]', clean_text)
    for m in cite_matches:
        cid = m.group(1).strip()
        stype = "statute" if "ra" in cid.lower() or "act" in cid.lower() else ("jurisprudence" if "g.r." in cid.lower() else "unknown")
        claims.append(Claim(
            text=f"Referenced citation {cid}",
            source_id=cid,
            quoted_text=None,
            source_type=stype
        ))

    return StructuredLegalResponse(claims=claims, answer_prose=clean_text)

def verify_citations_and_claims(
    response: StructuredLegalResponse,
    retrieved_chunks: List[Dict[str, Any]],
    query: str = "",
    min_fuzzy_ratio: float = 0.88,
    log_failures: bool = True
) -> VerificationSummary:
    """
    Two-Stage Citation & Claim Verification:
    Stage 1: Strict Source Existence Check (verifies source_id exists in retrieved turn context)
    Stage 2: Verbatim Quoted Text Fuzzy Verification (Levenshtein/partial ratio >= 0.88 if quote present)
    """
    total = len(response.claims)
    verified_count = 0
    failures = []
    annotated_prose = response.answer_prose

    # Build comprehensive lookup map from retrieved turn context
    chunk_map = {}
    for idx, chunk in enumerate(retrieved_chunks):
        doc_id = str(chunk.get("doc_id") or "").strip().lower()
        gr_no = str(chunk.get("gr_no") or "").strip().lower()
        title = str(chunk.get("title") or "").strip().lower()
        law_no = str(chunk.get("law_no") or "").strip().lower()
        norm_text = normalize_text_for_matching(chunk.get("text", ""))

        chk_entry = {"index": idx, "chunk": chunk, "norm_text": norm_text}
        chunk_map[f"source {idx+1}"] = chk_entry
        chunk_map[f"source_{idx+1}"] = chk_entry
        chunk_map[f"source{idx+1}"] = chk_entry
        chunk_map[f"doc {idx+1}"] = chk_entry
        chunk_map[f"doc_{idx+1}"] = chk_entry
        chunk_map[f"doc{idx+1}"] = chk_entry
        chunk_map[str(idx+1)] = chk_entry

        if doc_id: chunk_map[doc_id] = chk_entry
        if gr_no:
            chunk_map[gr_no] = chk_entry
            clean_gr = re.sub(r'[^a-z0-9]', '', gr_no)
            if clean_gr: chunk_map[clean_gr] = chk_entry
        if law_no: chunk_map[law_no] = chk_entry
        if title: chunk_map[title] = chk_entry

    retrieved_summary_ids = [c.get("gr_no") or c.get("title") or c.get("doc_id") for c in retrieved_chunks]

    for claim in response.claims:
        target_id = str(claim.source_id).strip().lower()
        clean_id = re.sub(r'[^a-z0-9]', '', target_id)

        matched_entry = None
        if target_id in chunk_map:
            matched_entry = chunk_map[target_id]
        elif clean_id in chunk_map:
            matched_entry = chunk_map[clean_id]
        else:
            for k, v in chunk_map.items():
                if target_id in k or (clean_id and clean_id in k):
                    matched_entry = v
                    break

        # Stage 1: Check Source Existence in Retrieved Turn Chunks
        if not matched_entry:
            claim.verified = False
            claim.failure_type = "source_not_retrieved"
            claim.verification_reason = f"Source ID '{claim.source_id}' was not retrieved in the active context for this turn"
            failures.append({
                "query": query,
                "claim_text": claim.text,
                "attempted_source_id": claim.source_id,
                "quoted_text": claim.quoted_text,
                "failure_type": "source_not_retrieved",
                "retrieved_source_ids": retrieved_summary_ids
            })
            continue

        # Stage 2: Fuzzy Verification of Quoted Excerpt (if present)
        if claim.quoted_text and claim.quoted_text.strip():
            norm_quote = normalize_text_for_matching(claim.quoted_text)
            norm_source = matched_entry["norm_text"]

            sim_score = fuzz.partial_ratio(norm_quote, norm_source) / 100.0
            claim.similarity_score = sim_score

            if sim_score < min_fuzzy_ratio:
                claim.verified = False
                claim.failure_type = "quote_mismatch"
                claim.verification_reason = f"Quoted text similarity ({sim_score:.2f}) below threshold ({min_fuzzy_ratio:.2f})"
                failures.append({
                    "query": query,
                    "claim_text": claim.text,
                    "attempted_source_id": claim.source_id,
                    "quoted_text": claim.quoted_text,
                    "similarity_score": sim_score,
                    "failure_type": "quote_mismatch",
                    "retrieved_source_ids": retrieved_summary_ids
                })
                continue

        claim.verified = True
        claim.failure_type = None
        claim.verification_reason = "Verified against retrieved turn context"
        verified_count += 1

    # Annotate Prose with Specific Failure Badges
    for claim in response.claims:
        if claim.verified is False:
            if claim.failure_type == "source_not_retrieved":
                badge_label = f"⚠️ [{claim.source_id} (Unverified — Not Retrieved)]"
            elif claim.failure_type == "quote_mismatch":
                badge_label = f"⚠️ [{claim.source_id} (Unverified — Quote Mismatch)]"
            else:
                badge_label = f"⚠️ [{claim.source_id} (Unverified)]"

            flag_badge = f'<span class="unverified-citation text-rose-600 font-bold" title="[UNVERIFIED: {claim.verification_reason}]">{badge_label}</span>'
            pattern = r'\[\s*' + re.escape(claim.source_id) + r'\s*\]'
            annotated_prose = re.sub(
                pattern,
                flag_badge,
                annotated_prose,
                flags=re.IGNORECASE
            )

    accuracy = (verified_count / total) if total > 0 else 1.0

    if log_failures and failures:
        try:
            os.makedirs(os.path.dirname(VERIFICATION_LOG_PATH), exist_ok=True)
            with open(VERIFICATION_LOG_PATH, "a", encoding="utf-8") as f:
                for fail in failures:
                    fail["timestamp"] = datetime.utcnow().isoformat()
                    f.write(json.dumps(fail, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed writing to {VERIFICATION_LOG_PATH}: {e}")

    return VerificationSummary(
        total_claims=total,
        verified_claims=verified_count,
        unverified_claims=total - verified_count,
        accuracy_rate=accuracy,
        failures=failures,
        annotated_prose=annotated_prose
    )
