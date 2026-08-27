# doctrine_currency.py - Philippine Supreme Court Landmark Precedent Currency Engine
"""
=============================================================================
PHILIPPINE SUPREME COURT DOCTRINE CURRENCY & CITATION TREATMENT ENGINE
=============================================================================

This module maintains the ground-truth doctrinal currency status of Philippine
Supreme Court jurisprudence. It tracks whether seminal precedents are [GOOD LAW],
[MODIFIED], [REVERSED], [ABANDONED], or [SUPERSEDED] by subsequent En Banc rulings.

UPDATE PROCESS & TRIAGE PROTOCOL:
1. Daily / Weekly Triage:
   - When new En Banc decisions are promulgated by the Supreme Court of the
     Philippines (sc.judiciary.gov.ph), check if they explicitly modify, overturn,
     or recalibrate prior doctrines (e.g., under Family Code Art. 36, Cybercrime
     libel, condonation doctrine, or custodial investigation rules).
2. Knowledge Base Update:
   - Add or update the G.R. number entry in `KNOWN_DOCTRINE_STATUS`.
   - Set `status`, `last_verified_date` (ISO YYYY-MM-DD), and `source_basis`:
     * "manually_curated": Verified directly by legal researcher / jurisprudence review.
     * "derived_from_citation_treatment": Parsed from SC syllabus Shepardizing text.
     * "pending_review": Newly flagged decision awaiting senior legal review.
3. Staleness Warning Trigger:
   - If `last_verified_date` is older than `DOCTRINE_STALENESS_THRESHOLD_DAYS` (180 days / 6 months),
     the system automatically appends a soft currency warning (e.g., `[GOOD LAW — status as of May 2024]`).
"""

import os
import re
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CaseStatus = Literal["statute", "good_law", "reversed", "modified", "abandoned", "superseded", "unknown"]
SourceBasis = Literal["manually_curated", "derived_from_citation_treatment", "pending_review"]

DOCTRINE_STALENESS_THRESHOLD_DAYS = int(os.getenv("DOCTRINE_STALENESS_DAYS", "180"))

class DoctrineRecord(BaseModel):
    gr_no: str
    case_title: str
    status: CaseStatus
    last_verified_date: str = Field(description="ISO Date YYYY-MM-DD when doctrine currency was last verified")
    source_basis: SourceBasis = "manually_curated"
    ponente: str = ""
    notes: str = ""

# Curated Registry of Philippine Landmark Doctrines with Versioning & Verification Metadata
KNOWN_DOCTRINE_STATUS: Dict[str, DoctrineRecord] = {
    # G.R. No. 108763: Republic v. CA & Molina (1997) -> Second guideline modified by Tan-Andal
    "108763": DoctrineRecord(
        gr_no="108763",
        case_title="Republic v. Court of Appeals and Molina",
        status="modified",
        last_verified_date="2026-05-15",
        source_basis="manually_curated",
        ponente="Panganiban, J.",
        notes="Second guideline on mandatory medical/psychiatric examination modified by Tan-Andal (2021)."
    ),
    # G.R. No. 196359: Tan-Andal v. Andal (2021, En Banc) -> Current Controlling Law on Art. 36
    "196359": DoctrineRecord(
        gr_no="196359",
        case_title="Tan-Andal v. Andal",
        status="good_law",
        last_verified_date="2026-06-01",
        source_basis="manually_curated",
        ponente="Leonen, J.",
        notes="En Banc landmark ruling redefining psychological incapacity as a legal concept proven by totality of evidence."
    ),
    # G.R. No. 112019: Santos v. CA (1995) -> Early landmark on psychological incapacity
    "112019": DoctrineRecord(
        gr_no="112019",
        case_title="Santos v. Court of Appeals",
        status="modified",
        last_verified_date="2025-08-10",
        source_basis="manually_curated",
        ponente="Vitug, J.",
        notes="Requisites of gravity, antecedence, and incurability modified in application by Tan-Andal."
    ),
    # G.R. No. 119190: Chi Ming Tsoi v. CA (1997) -> Continuous refusal of marital intercourse
    "119190": DoctrineRecord(
        gr_no="119190",
        case_title="Chi Ming Tsoi v. Court of Appeals",
        status="good_law",
        last_verified_date="2026-04-12",
        source_basis="manually_curated",
        ponente="Torres, Jr., J.",
        notes="Senseless refusal to consummate marriage constitutes psychological incapacity."
    ),
    # G.R. No. 203335: Disini v. Secretary of Justice (2014, En Banc) -> Cybercrime Prevention Act
    "203335": DoctrineRecord(
        gr_no="203335",
        case_title="Disini v. Secretary of Justice",
        status="good_law",
        last_verified_date="2026-03-20",
        source_basis="manually_curated",
        ponente="Abad, J.",
        notes="Upheld constitutionality of Section 4(c)(4) online libel against initial perpetrators."
    ),
    # G.R. No. 217126: Carpio-Morales v. CA (2015, En Banc) -> Abandonment of Condonation Doctrine
    "217126": DoctrineRecord(
        gr_no="217126",
        case_title="Carpio-Morales v. Court of Appeals (Binay Jr.)",
        status="good_law",
        last_verified_date="2026-02-14",
        source_basis="manually_curated",
        ponente="Perlas-Bernabe, J.",
        notes="Expressly abandoned the Aguinaldo condonation doctrine for elective public officials prospectively."
    ),
    # G.R. No. 221029: Republic v. Manalo (2018, En Banc) -> Foreign Divorce under Art. 26(2)
    "221029": DoctrineRecord(
        gr_no="221029",
        case_title="Republic v. Manalo",
        status="good_law",
        last_verified_date="2026-01-18",
        source_basis="manually_curated",
        ponente="Peralta, J.",
        notes="Filipino spouse may remarry even if foreign divorce was initiated by the Filipino spouse."
    ),
    # G.R. No. 81561: Valmonte v. De Villa (1989) -> Constitutionality of Checkpoints
    "81561": DoctrineRecord(
        gr_no="81561",
        case_title="Valmonte v. De Villa",
        status="good_law",
        last_verified_date="2025-01-10",  # Older than 180 days to verify staleness soft warning
        source_basis="manually_curated",
        ponente="Padilla, J.",
        notes="Routine military/police checkpoints conducted properly do not violate constitutional protections against warrantless searches."
    ),
    # G.R. No. 170470: People v. Malngan (2006) -> Custodial Investigation admissions
    "170470": DoctrineRecord(
        gr_no="170470",
        case_title="People v. Malngan",
        status="good_law",
        last_verified_date="2026-04-05",
        source_basis="manually_curated",
        ponente="Chico-Nazario, J.",
        notes="Un-counseled extrajudicial confessions during custodial investigation are inadmissible."
    )
}

HISTORICAL_QUERY_PATTERNS = [
    r"\bhistorical\b", r"\bhistory\b", r"\boverruled\b", r"\babandoned\b",
    r"\bbefore\b", r"\bprevious\b", r"\bpreviously\b", r"\bold doctrine\b", r"\bformer\b",
    r"dating doktrina", r"nakaraan", r"bago ang", r"prior to"
]

def is_historical_query(query: str) -> bool:
    if not query:
        return False
    q_lower = query.lower()
    return any(re.search(pattern, q_lower) for pattern in HISTORICAL_QUERY_PATTERNS)

def get_case_record(doc: Dict[str, Any]) -> Optional[DoctrineRecord]:
    """Retrieves full DoctrineRecord metadata if G.R. docket matches curated knowledge base."""
    gr_no = str(doc.get("gr_no") or "").lower()
    digits_match = re.search(r'(\d{5,7})', gr_no)
    if digits_match:
        gr_digits = digits_match.group(1)
        if gr_digits in KNOWN_DOCTRINE_STATUS:
            return KNOWN_DOCTRINE_STATUS[gr_digits]
    return None

def get_case_status(doc: Dict[str, Any]) -> str:
    cat = str(doc.get("category") or "").lower()
    if "republic act" in cat or "statute" in cat or "bill" in cat:
        return str(doc.get("status") or "good_law")

    explicit_status = doc.get("status")
    if explicit_status and explicit_status in ["good_law", "reversed", "modified", "abandoned", "superseded", "unknown"]:
        return explicit_status

    rec = get_case_record(doc)
    if rec:
        return rec.status

    text_sample = (str(doc.get("summary") or "") + " " + str(doc.get("text") or "")).lower()
    if "expressly overruled" in text_sample or "doctrine is hereby abandoned" in text_sample:
        return "abandoned"
    elif "modified the guidelines" in text_sample or "recalibrated" in text_sample:
        return "modified"

    return "unknown"

def get_formatted_doctrine_badge(doc: Dict[str, Any], max_age_days: int = DOCTRINE_STALENESS_THRESHOLD_DAYS) -> str:
    """
    Renders UI doctrine badge. If last_verified_date > max_age_days,
    surfaces soft staleness warning: '[GOOD LAW — status as of MMM YYYY]'.
    """
    status = get_case_status(doc)
    rec = get_case_record(doc)

    if not rec or not rec.last_verified_date:
        if status == "good_law": return "✓ GOOD LAW"
        if status == "modified": return "⚠️ MODIFIED"
        if status in ("reversed", "abandoned"): return "❌ OVERRULED"
        return "UNKNOWN STATUS"

    try:
        ver_date = datetime.strptime(rec.last_verified_date, "%Y-%m-%d").date()
        today = date.today()
        age_days = (today - ver_date).days
        formatted_month_year = ver_date.strftime("%b %Y")

        if age_days > max_age_days:
            # Stale verification: surface soft warning with date
            if status == "good_law":
                return f"✓ GOOD LAW (verified as of {formatted_month_year})"
            elif status == "modified":
                return f"⚠️ MODIFIED (verified as of {formatted_month_year})"
            elif status in ("reversed", "abandoned"):
                return f"❌ OVERRULED (verified as of {formatted_month_year})"
        else:
            if status == "good_law": return "✓ GOOD LAW"
            if status == "modified": return "⚠️ MODIFIED"
            if status in ("reversed", "abandoned"): return "❌ OVERRULED"
    except Exception as e:
        logger.debug(f"Error parsing last_verified_date {rec.last_verified_date}: {e}")

    return "✓ GOOD LAW" if status == "good_law" else status.upper()

def filter_and_tag_doctrine_currency(docs: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    is_hist = is_historical_query(query)
    tagged_docs = []

    for doc in docs:
        status = get_case_status(doc)
        rec = get_case_record(doc)
        doc["doctrine_status"] = status
        doc["doctrine_badge"] = get_formatted_doctrine_badge(doc)
        if rec:
            doc["last_verified_date"] = rec.last_verified_date
            doc["source_basis"] = rec.source_basis
            doc["doctrine_notes"] = rec.notes
        tagged_docs.append(doc)

    if not is_hist:
        valid_docs = [d for d in tagged_docs if d["doctrine_status"] not in ("reversed", "abandoned")]
        bad_docs = [d for d in tagged_docs if d["doctrine_status"] in ("reversed", "abandoned")]
        return valid_docs if valid_docs else bad_docs

    return tagged_docs
