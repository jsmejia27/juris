# tests/test_research_tracks.py
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_pipeline import (
    get_prompt_for_track,
    build_timeline_data,
    PROMPT_TAB1_TREATISE,
    PROMPT_EXECUTIVE_BRIEF,
    PROMPT_CORPORATE_ADVISORY,
    PROMPT_BAR_ACADEMIC
)
from server import ChatRequest

def test_prompt_selection_for_tracks():
    assert get_prompt_for_track("treatise") == PROMPT_TAB1_TREATISE
    assert get_prompt_for_track("executive") == PROMPT_EXECUTIVE_BRIEF
    assert get_prompt_for_track("corporate") == PROMPT_CORPORATE_ADVISORY
    assert get_prompt_for_track("bar") == PROMPT_BAR_ACADEMIC
    # Fallback to default
    assert get_prompt_for_track("unknown_track") == PROMPT_TAB1_TREATISE
    assert get_prompt_for_track(None) == PROMPT_TAB1_TREATISE

def test_build_timeline_data_sorting_and_formatting():
    sources = [
        {
            "title": "Bustamante v. NLRC",
            "gr_no": "G.R. No. 111651",
            "year": 1996,
            "doctrine_status": "Overruled Precedent",
            "category": "Supreme Court Jurisprudence",
            "summary": "Full backwages doctrine formulation.",
            "text": "The Court en banc establishes full backwages..."
        },
        {
            "title": "Republic Act No. 6715",
            "law_no": "RA 6715",
            "date": "March 2, 1989",
            "category": "Statute",
            "summary": "Herrera-Veloso Law amending the Labor Code."
        },
        {
            "title": "Mercury Drug v. Court of Industrial Relations",
            "gr_no": "G.R. No. L-23357",
            "date": "April 30, 1974",
            "doctrine_status": "Overruled",
            "summary": "Three-year backwages cap rule."
        }
    ]

    timeline = build_timeline_data(sources)
    assert len(timeline) == 3
    # Check chronological order (1974 -> 1989 -> 1996)
    assert timeline[0]["year"] == 1974
    assert "Mercury Drug" in timeline[0]["title"]
    assert timeline[1]["year"] == 1989
    assert timeline[2]["year"] == 1996
    assert timeline[2]["status"] == "Overruled Precedent"

def test_build_timeline_data_empty():
    assert build_timeline_data([]) == []
    assert build_timeline_data(None) == []

def test_chat_request_schema():
    req = ChatRequest(
        message="What are the grounds for legal separation under Article 55?",
        research_track="corporate"
    )
    assert req.research_track == "corporate"
    assert req.model == "qwen3.5:9b"

    default_req = ChatRequest(message="Test query")
    assert default_req.research_track == "treatise"
