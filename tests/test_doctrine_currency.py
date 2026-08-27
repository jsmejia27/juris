import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from doctrine_currency import (
    get_case_status,
    is_historical_query,
    filter_and_tag_doctrine_currency
)

def test_doctrine_status_tan_andal():
    doc_tan = {"gr_no": "G.R. No. 196359", "category": "Jurisprudence"}
    assert get_case_status(doc_tan) == "good_law"

def test_doctrine_status_molina():
    doc_molina = {"gr_no": "G.R. No. 108763", "category": "Jurisprudence"}
    assert get_case_status(doc_molina) == "modified"

def test_doctrine_status_statute():
    doc_ra = {"gr_no": "RA 9262", "category": "Republic Act"}
    assert get_case_status(doc_ra) == "good_law"

def test_historical_query_detection():
    assert is_historical_query("What was the old doctrine before Tan-Andal?") == True
    assert is_historical_query("Ano ang dating doktrina sa artikulo 36?") == True
    assert is_historical_query("What are the current requisites under Article 36?") == False

def test_filter_abandoned_cases():
    docs = [
        {"title": "Case A (Good )", "gr_no": "G.R. 196359", "category": "Jurisprudence"},
        {"title": "Case B (Abandoned)", "gr_no": "G.R. 999999", "status": "abandoned", "category": "Jurisprudence"}
    ]
    #Non-historical query: abandoned document is filtered out
    results = filter_and_tag_doctrine_currency(docs, query="What is the ruling on Art. 36?")
    assert len(results) == 1
    assert results[0]["gr_no"] == "G.R. 196359"
    assert results[0]["doctrine_status"] == "good_law"

    #Historical query: both documents preserved with status tags
    hist_results = filter_and_tag_doctrine_currency(docs, query="What was the abandoned doctrine in past cases?")
    assert len(hist_results) == 2