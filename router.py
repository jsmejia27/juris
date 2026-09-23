import os, re, json, logging, threading
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

logger = logging.getLogger(__name__)

ComplexityLevel = Literal[
    "DIRECT_LOOKUP", "STATUTORY_ANALYSIS", "MULTI_DOCTRINE_SYNTHESIS",
    "CONFLICT_OF_LAWS", "CONSTITUTIONAL_REVIEW", "OUT_OF_SCOPE"
]

ROUTER_LOG_PATH = "logs/routing_decisions.jsonl"

CONFLICT_PATTERNS = [
    r"\bconflict\b", r"\binconsistent\b", r"\brepealed by implication\b",
    r"\bprevails over\b", r"\bhierarchy of laws\b", r"\bspecial law vs general\b",
    r"\bnagkakasalungat\b", r"\bnaggugulo\b", r"\bano ang masusunod\b"
]

MULTI_DOCTRINE_PATTERNS = [
    r"\bevolution of doctrine\b", r"\blandmark precedents?\b", r"\bhistory of rulings\b",
    r"\bcompare\b", r"\bcontrolling over\b", r"\btan-andal.*molina\b", r"\bmolina.*tan-andal\b",
    r"\bmultiple decisions\b", r"\bcross-statute\b", r"\binterplay between\b",
    r"\boverruled?\b", r"\bdoctrinal shift\b", r"\breconcil(?:e|ing)\b", r"\bmodified\b"
]

CONSTITUTIONAL_PATTERNS = [
    r"\bunconstitutional\b", r"\bequal protection clause\b", r"\bdue process\b",
    r"\bbill of rights\b", r"\bseparation of powers\b", r"\bpolice power\b",
    r"\bjudicial review\b", r"\bconstitutional challenge\b", r"\blimitation on rights\b"
]

PENDING_BILLS_PATTERNS = [
    r"\b(?:senate|house)\s*bill\b", r"\b(?:sb|hb)\s*(?:no\.?)?\s*\d+\b",
    r"\bpending\s*(?:bill|legislation|law)\b", r"\bdivorce\s*bill\b",
    r"\bsogie\s*bill\b", r"\bproposed\s*(?:bill|law|measure)\b"
]

class LayaLegalRouter:
    """
    Non-autoregressive System 1 decision engine based on Laya (ModernBERT-RLCD).
    Executes multi-class complexity classification, out-of-scope guardrails,
    and pending legislation routing in single forward passes.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(LayaLegalRouter, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, preload: bool = True):
        if getattr(self, "_initialized", False):
            return
        self.router = None
        self.available = False
        try:
            import torch
            from laya import Router
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing Laya Decision Engine on {device} (preload={preload})...")
            self.router = Router(preload=preload, device=device)
            self.available = True
            logger.info("Laya Decision Engine successfully initialized.")
        except Exception as e:
            logger.warning(f"Laya neural router could not be loaded: {e}. Falling back to regex heuristics.")
            self.router = None
            self.available = False

        self.questions = {
            "complexity": {
                "type": "choice",
                "instructions": "Classify the legal reasoning complexity level of this question:",
                "criteria": {
                    "STATUTORY_ANALYSIS": "Questions asking about legal requisites, elements, prescriptive periods, validity, grounds, definitions, or rights under laws and jurisprudence",
                    "DIRECT_LOOKUP": "Direct statutory questions on specific penalties, leaves, cash subsidies, benefits, or provisions in a Republic Act or Article",
                    "MULTI_DOCTRINE_SYNTHESIS": "Case doctrines, comparing judicial rulings, Supreme Court decisions, landmark precedents like Tan-Andal and Molina",
                    "CONFLICT_OF_LAWS": "Conflict between two laws, hierarchy of laws, repeal by implication, special vs general law",
                    "CONSTITUTIONAL_REVIEW": "Constitutional challenges, due process, equal protection, police power, or bill of rights",
                    "NON_LEGAL": "Completely non-legal topics such as computer programming, cooking recipes, calculus, or hardware"
                }
            },
            "needs_open_congress": {
                "type": "choice",
                "instructions": "Does this question ask about pending legislative bills or proposed measures in Congress?",
                "criteria": {
                    "yes": "Questions explicitly asking about pending bills, Senate bills, House bills, proposed legislation, or legislative status in Congress",
                    "no": "Questions regarding enacted Republic Acts, Supreme Court jurisprudence, or general legal doctrine"
                }
            }
        }
        self._initialized = True

    def classify(self, query: str) -> Optional[Dict[str, Any]]:
        if not self.available or not self.router or not query:
            return None
        try:
            state = {"text": query}
            res = self.router.predict(state, self.questions)
            answers = res.get("answers", {})

            complexity_ans = answers.get("complexity", {})
            complexity = complexity_ans.get("choice", "STATUTORY_ANALYSIS")
            conf = complexity_ans.get("confidence", 0.85)

            needs_bills = answers.get("needs_open_congress", {}).get("choice") == "yes"

            q_lower = query.lower()
            has_legal_anchors = any(
                term in q_lower for term in [
                    "ra ", "republic act", "g.r.", "article", "section", "court",
                    "jurisprudence", "libel", "incapacity", "pilipinas", "philippines",
                    " v. ", " versus ", "law", "batasan", "senate bill", "house bill",
                    "parusa", "prescriptive", "prescribe", "requisite", "element"
                ]
            )

            if has_legal_anchors and complexity == "NON_LEGAL":
                complexity = "STATUTORY_ANALYSIS"

            is_ph = (complexity != "NON_LEGAL") or has_legal_anchors

            return {
                "complexity": "DIRECT_LOOKUP" if complexity == "NON_LEGAL" else complexity,
                "score": float(conf) if conf is not None else 0.85,
                "reasons": [f"Laya neural classification ({complexity})"],
                "is_philippine_law": is_ph,
                "needs_open_congress": needs_bills,
                "routing_engine": "laya_modernbert"
            }
        except Exception as e:
            logger.debug(f"Laya inference exception: {e}")
            return None

_global_laya_router: Optional[LayaLegalRouter] = None

def get_laya_router() -> LayaLegalRouter:
    global _global_laya_router
    if _global_laya_router is None:
        _global_laya_router = LayaLegalRouter(preload=True)
    return _global_laya_router

def classify_legal_query_regex(query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Regex heuristic classifier fallback.
    """
    if not query:
        return {
            "complexity": "DIRECT_LOOKUP",
            "score": 0.0,
            "reasons": ["Empty query"],
            "is_philippine_law": True,
            "needs_open_congress": False,
            "routing_engine": "regex_heuristic"
        }

    q_lower = query.lower()
    needs_bills = any(re.search(p, q_lower) for p in PENDING_BILLS_PATTERNS)

    # 1. Check Conflict of Laws / Statutory Interplay
    if any(re.search(p, q_lower) for p in CONFLICT_PATTERNS):
        return {
            "complexity": "CONFLICT_OF_LAWS",
            "score": 0.90,
            "reasons": ["Matched conflict of laws / statutory interplay patterns"],
            "is_philippine_law": True,
            "needs_open_congress": needs_bills,
            "routing_engine": "regex_heuristic"
        }

    # 2. Check Multi-Doctrine Synthesis
    if any(re.search(p, q_lower) for p in MULTI_DOCTRINE_PATTERNS):
        return {
            "complexity": "MULTI_DOCTRINE_SYNTHESIS",
            "score": 0.85,
            "reasons": ["Matched multi-doctrine / judicial evolution patterns"],
            "is_philippine_law": True,
            "needs_open_congress": needs_bills,
            "routing_engine": "regex_heuristic"
        }

    # 3. Check Constitutional Review
    if any(re.search(p, q_lower) for p in CONSTITUTIONAL_PATTERNS):
        return {
            "complexity": "CONSTITUTIONAL_REVIEW",
            "score": 0.80,
            "reasons": ["Matched Constitutional / Bill of Rights patterns"],
            "is_philippine_law": True,
            "needs_open_congress": needs_bills,
            "routing_engine": "regex_heuristic"
        }

    # 4. Check if multiple RAs or G.R. numbers are mentioned
    ra_count = len(re.findall(r'\b(?:republic\s+act|ra)\s*(?:no\.?)?\s*\d+\b', q_lower))
    gr_count = len(re.findall(r'\bg\.r\.\s*(?:no\.?)?\s*\d+\b', q_lower))
    if ra_count + gr_count > 1:
        return {
            "complexity": "MULTI_DOCTRINE_SYNTHESIS",
            "score": 0.75,
            "reasons": ["Multiple statutory/docket references detected"],
            "is_philippine_law": True,
            "needs_open_congress": needs_bills,
            "routing_engine": "regex_heuristic"
        }

    # 5. Statutory analysis vs Direct lookup
    if len(query.split()) > 12 or any(w in q_lower for w in ["explain", "how does", "paano", "requisites", "elements"]):
        return {
            "complexity": "STATUTORY_ANALYSIS",
            "score": 0.60,
            "reasons": ["Complex explanatory legal query"],
            "is_philippine_law": True,
            "needs_open_congress": needs_bills,
            "routing_engine": "regex_heuristic"
        }

    return {
        "complexity": "DIRECT_LOOKUP",
        "score": 0.30,
        "reasons": ["Standard statutory/case lookup"],
        "is_philippine_law": True,
        "needs_open_congress": needs_bills,
        "routing_engine": "regex_heuristic"
    }

def classify_legal_query(query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Main classifier entrypoint: Attempts Laya non-autoregressive neural classification first,
    falling back to regex heuristics if unavailable.
    """
    router = get_laya_router()
    res = router.classify(query)
    if res is not None:
        return res
    return classify_legal_query_regex(query, history)

def route_query(query: str, history: Optional[List[Dict[str, str]]] = None, local_model: str = "qwen3.5:9b") -> Dict[str, Any]:
    """
    Routes the legal query to the optimal model and execution tier.
    """
    classification = classify_legal_query(query, history)
    complexity = classification.get("complexity", "DIRECT_LOOKUP")
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
            chosen_model = "qwen3:14b" if local_model == "qwen3:14b" else "qwen3.5:9b"
            routing_type = "local_fallback"

    decision_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "complexity": complexity,
        "score": classification.get("score", 0.85),
        "reasons": classification.get("reasons", []),
        "is_philippine_law": classification.get("is_philippine_law", True),
        "needs_open_congress": classification.get("needs_open_congress", False),
        "routing_engine": classification.get("routing_engine", "laya_modernbert"),
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
