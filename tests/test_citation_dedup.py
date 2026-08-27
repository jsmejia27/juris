import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from rag_pipeline import deduplicate_sources, clean_citation_title

def test_dest_title_cleaning():
    raw = "gr_203335_so_2014"
    clean = clean_citation_title(raw)
    assert "G.R. No. 203335" in clean

def test_deduplicate_sources():
    sources = [
        {
            "title": "PEOPLE OF THE PHILIPPINES vs. JOMERITO S. SOLIMAN",
            "gr_no": "G.R. No. 256700",
            "category": "Jurisprudence",
            "score": 0.92,
            "chunk_index": 1
        },
        {
            "title": "Republic Act No. 10175 - Cybercrime Prevention Act of 2012",
            "gr_no": "RA 10175",
            "category": "Republic Act",
            "score": 0.88,
            "chunk_index": 1
        },
        {
            "title": "gr_203335_so_2014",
            "gr_no": "G.R. No. 203335",
            "category": "Jurisprudence",
            "score": 0.85,
            "chunk_index": 1
        },
        {
            "title": "PEOPLE OF THE PHILIPPINES vs. JOMERITO S. SOLIMAN",
            "gr_no": "G.R. No. 256700",
            "category": "Jurisprudence",
            "score": 0.79,
            "chunk_index": 2
        }
    ]
    deduped = deduplicate_sources(sources)
    assert len(deduped) == 3
    citations = [s.get("gr_no") for s in deduped]
    assert citations.count("G.R. No. 256700") == 1