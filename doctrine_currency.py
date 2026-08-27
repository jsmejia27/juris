#breakthrough doctrine_currency.py
import re
import logging
from typing import List, Dict, Any, Literal

logger = logging.getLogger(__name__)

CaseStatus = Literal["statute", "good_law", "reversed", "modified", "abandoned", "superseded", "unknown"]

HISTORICAL_QUERY_PATTERNS = [
    r"\bhistorical\b", r"\bhistory\b", r"\boverruled\b", r"\babandoned\b",
    r"\bbefore\b", r"\bprevious\b", r"\bpreviously\b", r"\bold doctrine\b", r"\bformer\b",
    r"dating doktrina", r"nakaraan", r"bago ang", r"prior to"
]

# Database of known Philippine landmark doctrine treatments
KNOWN_DOCTRINE_STATUS: Dict[str, CaseStatus] = {
    #108763: Republic v. CA & Molina (G.R. No. 108763, 1997) -> Modified by Tan-Andal
    "108763": "modified",
    #196359: Tan-Andal v. Andal (G.R. No. 196359, 2021) -> Current Good Law
    "196359": "good_law",
    #112019: Santos v. CA (G.R. No. 112019) -> Modified
    "112019": "modified",
    #119190: Chi Ming Tsoi v. CA (G.R. No. 119190, Sexual Destitution)
    "119190": "good_law",
    #203335: Disini v. Secretary of Justice (G.R. No. 203335, Cybercrime Prevention Act)
    "203335": "good_law",
    #170470: People v. Malngan
    "170470": "good_law",
    #81561: Valmonte v. De Villa (Checkpoints)
    "81561": "good_law"
}

def is_historical_query(query: str) -> bool:
    if not query:
        return False
    q_lower = query.lower()
    return any(re.search(pattern, q_lower) for pattern in HISTORICAL_QUERY_PATTERNS)

def get_case_status(doc: Dict[str, Any]) -> str:
    cat = str(doc.get("category") or "").lower()
    if "republic act" in cat or "statute" in cat or "bill" in cat:
        return str(doc.get("status") or "good_law")

    explicit_status = doc.get("status")
    if explicit_status and explicit_status in ["good_law", "reversed", "modified", "abandoned", "superseded", "unknown"]:
        return explicit_status

    gr_no = str(doc.get("gr_no") or "").lower()
    digits_match = re.search(r'(\d{5,7})', gr_no)
    if digits_match:
        gr_digits = digits_match.group(1)
        if gr_digits in KNOWN_DOCTRINE_STATUS:
            return KNOWN_DOCTRINE_STATUS[gr_digits]

    text_sample = (str(doc.get("summary") or "") + " " + str(doc.get("text") or "")).lower()
    if "expressly overruled" in text_sample or "doctrine is hereby abandoned" in text_sample:
        return "abandoned"
    elif "modified the guidelines" in text_sample or "recalibrated" in text_sample:
        return "modified"

    return "unknown"

def filter_and_tag_doctrine_currency(docs: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    is_hist = is_historical_query(query)
    tagged_docs = []

    for doc in docs:
        status = get_case_status(doc)
        doc["doctrine_status"] = status
        tagged_docs.append(doc)

    if not is_hist:
        valid_docs = [d for d in tagged_docs if d["doctrine_status"] not in ("reversed", "abandoned")]
        bad_docs = [d for d in tagged_docs if d["doctrine_status"] in ("reversed", "abandoned")]
        return valid_docs if valid_docs else bad_docs

    return tagged_docs
