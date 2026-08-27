import os, re, json, logging
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

logger = logging.getLogger(__name__)

ComplexityLevel = Literal[
    "DIRECT_LOOKUP", "STATUTORY_ANALYSIS", "MULTI_DOCTRINE_SYNTHESIS",
    "CONFLICT_OF_LAWS", "CONSTITUTIONAL_REVIEW"
]

ROSTER_LOG_PATHs = "logs/routing_decisions.jsonl"

CONFLICT_PATTERNS = [
    r"\bconflict\b", r"\binconsistent\b", r"\brepealed by implication\b",
    r"\bprevails over\b", r"\bhierarchy of laws\b", r"\bspecial law vs general\b",
    r"\bnagkakasalungat\b", r"\bnaggugulo\b", r"\bano ang masusunod\b"
]

MULTI_DOCTRINE_PATTERNS = [
    r"\bevolution of doctrine\b", r"\blandmark precedents\b", r"\bhistory of rulings\b",
    r"\bcompare\b", r"\btan-andal vs molina\b", r"\bmolina and tan-andal\b",
    r"\bmultiple decisions\b", r"\bcross-statute\b", r"\binterplay between\b"
]

CONSTITUTIONAL_PATTERNS = [
    r"\bunconstitutional\b", r"\bequal protection clause\b", r"\bdue process\b",
    r"\bbill of rights\b", r"\bseparation of powers\b", r"\bpolice power\b",
    r"\bjudicial review\b", r"\bconstitutional challenge\b", r"\blimitation on rights\b"
]

def classify_legal_query(query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Classifies Philippine legal queries into specific complexity categories.
    """
    if not query:
        return {"complexity": "DIRECT_LOOKUP", "score": 0.0, "reasons": ["mta query"]}

    q_lower = query.lower()
    reasons = []

    # 1. Check Conflict of Laws / Statutory Interplay
    if any(re.search(p, q_lower) for p in CONFLICT_PATTERNS):
        return {
            "complexity": "CONFLICT_OF_LAWS",
            "score": 0.9,
            "reasons": ["Matched conflict of laws / statutory interplay patterns"]
        }

    # 2. Check Multi-Doctrine Synthesis
    if any(re.search(p, q_lower) for p in MULTI_DOCTRINE_PATTERNS):
        return {
            "complexity": "MULTI_DOCTRINE_SYNTHESIS",
            "score": 0.85,
            "reasons": ["Matched multi-doctrine / jduicial evolution patterns"]
        }

    # 3. Check Constitutional Review
    if any(re.search(p, q_lower) for p in CONSTITUTIONAL_PATTERNS):
        return {
            "complexity": "CONSTITUTIONAL_REVIEW",
            "score": 0.80,
            "reasons": ["Matched Constitutional / Bill of Rights patterns"]
        }

    # 4. Check if multiple RAs or G R numbers are mentioned
    ra_count = len(re.findall(r'\b(?:republic\s+act|ra)\s*(?:no\.?)\s*\d+\b', q_lower))
    gr_count = len(re.findall(r'\bg\.rL\.\s* ?\d+\b', q_lower))
    if ra_count + gr_count > 1:
        return {
            "complexity": "MULTI_DOCTRINE_SYNTHESIS",
            "score": 0.75,
            "reasons": ["Multiple statutory/docket references detected"]
        }

    # 5. Statutory analysis vs Direct lookup
    if len(query.split()) > 12 or "explain" in q_lower or "how does" in q_lower or "paano" in q_lower:
        return {
            "complexity": "STATUTORY_ANALYSIS",
            "score": 0.60,
            "reasons": ["Complex explanatory legal query"]
        }

    return {
        "complexity": "DIRECT_LOOKUP",
        "score": 0.30,
        "reasons": ["Standard statutory/case lookup"]
    }

def route_query(query: str, history: Optional[List[Dict[str, str]]] = None, local_model: str = "qwen3.5:9b") -> Dict[str, Any]:
    """
    Routes the legal query to the optimal model.
    If FEATURE_FRONTIER_ROUTER is enabled and complexity is high, routes to a frontier model.
    Otherwise routes to local qwen3.5:9b or qwen3:14b.
    """
    classification = classify_legal_query(query, history)
    complexity = classification["complexity"]
    feature_flag_enabled = bool(os.getenv("FEATURE_FRONTIER_ROUTER") == "1")

    chosen_model = local_model
    routing_type = "local_ollama"

    if feature_flag_enabled and complexity in ("CONFLICT_OF_LAWS", "MULTI_DOCTRINE_SYNTHESIS", "CONSTITUTIONAL_REVIEW"):
        if os.getenv("ANTHROPIC_API_KEY"):
            chosen_model = "claude-3-5-sonnet-20241022"
            routing_type = "frontier_anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            chosen_model = "gpt-4o"
            routing_type = "frontier_openai"
        else:
            # Graceful fallback to local 14b if available, else 9b
            chosen_model = "qwen3:14b" if local_model == "qwen3:14b" else "qwen3.5:9b"
            routing_type = "local_fallback"

    decision_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "complexity": complexity,
        "score": classification["score"],
        "reasons": classification["reasons"],
        "chosen_model": chosen_model,
        "routing_type": routing_type
    }

    os.makedirs("logs", exist_ok=True)
    try:
        with open(ROUTER_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision_data) + "\n")
    except Exception as e:
        logger.debug(f"Failed to log routing decision: {e}")

    return decision_data
