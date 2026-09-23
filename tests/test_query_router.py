import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from router import (
    classify_legal_query,
    classify_legal_query_regex,
    route_query,
    get_laya_router,
    LayaLegalRouter
)

def test_laya_router_singleton():
    r1 = get_laya_router()
    r2 = get_laya_router()
    assert r1 is r2
    assert isinstance(r1, LayaLegalRouter)

def test_direct_lookup():
    q = "What is RA 9262?"
    res = classify_legal_query(q)
    assert res["complexity"] in ("DIRECT_LOOKUP", "STATUTORY_ANALYSIS")
    assert res["is_philippine_law"] is True
    assert res["needs_open_congress"] is False

def test_multi_doctrine_synthesis():
    q = "Compare the evolution of doctrine between Molina and Tan-Andal under Article 36."
    res = classify_legal_query(q)
    assert res["complexity"] in ("MULTI_DOCTRINE_SYNTHESIS", "STATUTORY_ANALYSIS", "CONFLICT_OF_LAWS")
    assert res["is_philippine_law"] is True

def test_conflict_of_laws():
    q = "Which law prevails over the other when there is a conflict between a general law and a special law?"
    res = classify_legal_query(q)
    assert res["complexity"] in ("CONFLICT_OF_LAWS", "STATUTORY_ANALYSIS", "CONSTITUTIONAL_REVIEW")
    assert res["is_philippine_law"] is True

def test_taglish_legal_query():
    q = "Ano ang prescriptive period ng online libel sa Pilipinas kapag sa Facebook pinost?"
    res = classify_legal_query(q)
    assert res["is_philippine_law"] is True

def test_pending_bills_detection():
    q = "What is the legislative status of Senate Bill No. 1234 on Absolute Divorce?"
    res = classify_legal_query(q)
    assert res["is_philippine_law"] is True
    assert res["needs_open_congress"] is True

def test_regex_fallback():
    q = "Is there a conflict between RA 9262 and the Revised Penal Code?"
    res = classify_legal_query_regex(q)
    assert res["complexity"] == "CONFLICT_OF_LAWS"
    assert res["routing_engine"] == "regex_heuristic"
    assert res["is_philippine_law"] is True

def test_route_query_payload_structure():
    q = "What are the requisites of psychological incapacity in Tan-Andal v. Andal?"
    route = route_query(q)
    assert "complexity" in route
    assert "chosen_model" in route
    assert "routing_type" in route
    assert "is_philippine_law" in route
    assert "needs_open_congress" in route
    assert "routing_engine" in route
    assert route["is_philippine_law"] is True
