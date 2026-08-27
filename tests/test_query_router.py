import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from router import classify_legal_query, route_query

def test_direct_lookup():
    q = "What is RA 9262?"
    res = classify_legal_query(q)
    assert res["complexity"] == "DIRECT_LOOKUP"

def test_multi_doctrine_synthesis():
    q = "Compare the evolution of doctrine between Molina and Tan-Andal under Article 36."
    res = classify_legal_query(q)
    assert res["complexity"] == "MULTI_DOCTRINE_SYNTHESIS"

def test_conflict_of_laws():
    q = "Which law prevails over the other when there is a conflict between a general law and a special law?"
    res = classify_legal_query(q)
    assert res["complexity"] == "CONFLICT_OF_LAWS"

def test_constitutional_review():
    q = "Is offshore data surveillance a violation of the equal protection clause and due process under the Bill of Rights?"
    res = classify_legal_query(q)
    assert res["complexity"] == "CONSTITUTIONAL_REVIEW"

def test_route_query_default_local():
    q = "What are the requirements for solo parent leave?"
    route = route_query(q)
    assert route["chosen_model"] == "qwen3.5:9b"
    assert route["routing_type"] == "local_ollama"
