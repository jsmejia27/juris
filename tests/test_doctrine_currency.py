import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from doctrine_currency import (
    get_case_status,
    is_historical_query,
    filter_and_tag_doctrine_currency,
    get_formatted_doctrine_badge,
    KNOWN_DOCTRINE_STATUS
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
        {"gr_no": "G.R. No. 196359", "title": "Tan-Andal", "text": "..."},
        {"gr_no": "G.R. No. 999999", "title": "Overruled Case", "text": "This doctrine is hereby abandoned..."}
    ]
    res_modern = filter_and_tag_doctrine_currency(docs, "What is current rule?")
    assert len(res_modern) == 1
    assert res_modern[0]["gr_no"] == "G.R. No. 196359"

    res_hist = filter_and_tag_doctrine_currency(docs, "What was the previous historical rule?")
    assert len(res_hist) == 2

def test_doctrine_staleness_badge():
    # Valmonte is dated 2025-01-10 (older than 180 days)
    doc_valmonte = {"gr_no": "G.R. No. 81561", "title": "Valmonte v. De Villa", "text": "checkpoints"}
    badge_valmonte = get_formatted_doctrine_badge(doc_valmonte, max_age_days=180)
    assert "verified as of" in badge_valmonte

    # Tan-Andal is dated 2026-06-01 (fresh)
    doc_tan_andal = {"gr_no": "G.R. No. 196359", "title": "Tan-Andal", "text": "art 36"}
    badge_tan = get_formatted_doctrine_badge(doc_tan_andal, max_age_days=180)
    assert badge_tan == "✓ GOOD LAW"

def test_doctrine_record_metadata():
    rec = KNOWN_DOCTRINE_STATUS["196359"]
    assert rec.source_basis == "manually_curated"
    assert rec.ponente == "Leonen, J."
    assert rec.last_verified_date == "2026-06-01"