# tests/test_pleading_drafting.py
import pytest
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pleading_drafting_service import PleadingDraftingService, AVAILABLE_TEMPLATES

def test_available_templates():
    service = PleadingDraftingService()
    templates = service.list_templates()
    assert len(templates) >= 5
    template_ids = [t["id"] for t in templates]
    assert "rule_45_petition" in template_ids
    assert "motion_reconsideration" in template_ids
    assert "judicial_affidavit" in template_ids
    assert "nlrc_position_paper" in template_ids
    assert "legal_opinion_memo" in template_ids

def test_template_prompt_generation():
    service = PleadingDraftingService()
    data = {
        "court_branch": "RTC Branch 20, Makati City",
        "case_number": "Civil Case No. R-MKT-26-00412-CV",
        "movant": "Spouses Roberto and Teresa Tan",
        "adverse_party": "Metrobank Corp.",
        "assailed_order_date": "August 15, 2026",
        "grounds": "Grave abuse of discretion in denying motion to dismiss based on improper venue",
        "rebuttal_points": "The parties explicitly stipulated exclusive venue in the City of Manila under Section 4, Rule 4 of the Rules of Court."
    }
    mock_authorities = "[Authority: Polytrade v. Blanco (G.R. No. L-27033)]\nStipulation of exclusive venue must be respected."
    prompt = service._get_template_prompt("motion_reconsideration", data, mock_authorities)
    
    assert "MOTION FOR RECONSIDERATION" in prompt
    assert "Polytrade v. Blanco" in prompt
    assert "RTC Branch 20, Makati City" in prompt
    assert "Spouses Roberto and Teresa Tan" in prompt

def test_docx_export():
    service = PleadingDraftingService()
    sample_draft = """# REPUBLIC OF THE PHILIPPINES
## REGIONAL TRIAL COURT
### BRANCH 20, MAKATI CITY

**SPOUSES ROBERTO TAN, ET AL.**,
Plaintiffs,

- versus -

Civil Case No. R-MKT-26-00412-CV

**METROBANK CORP.**,
Defendant.
x---------------------------------------------x

# MOTION FOR RECONSIDERATION

COMES NOW the Defendant, through counsel, and respectfully moves for the reconsideration of the Honorable Court's Order:

### I. TIMELINESS OF THE MOTION
1. The assailed order was received on August 15, 2026.
2. Under Section 1, Rule 37 of the Rules of Court, defendant has 15 days to file this motion.

> "A stipulation on exclusive venue is binding and enforceable."

### PRAYER
WHEREFORE, premises considered, it is respectfully prayed that the Order be reconsidered and set aside.
"""
    docx_stream = service.export_to_docx(sample_draft, title="MR_Test")
    assert isinstance(docx_stream, io.BytesIO)
    content = docx_stream.getvalue()
    assert len(content) > 1000  # Non-empty valid DOCX zip binary
    assert content.startswith(b"PK") # Standard zip archive header for docx
