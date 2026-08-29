# tests/test_legal_ingestion_service.py
import pytest
from unittest.mock import patch, MagicMock
from legal_ingestion_service import LegalIngestionService

SAMPLE_LAWS_HTML = """
<html>
<head><title>Republic Act No. 12254</title></head>
<body>
<p><font size="1">The Lawphil Project - Arellano Law Foundation</font></p>
<center>
<h3>REPUBLIC ACT NO. 12254</h3>
<h4>AN ACT INSTITUTIONALIZING THE TRANSITION OF THE GOVERNMENT TO E-GOVERNANCE, STRENGTHENING THE ICT ACADEMY, AND APPROPRIATING FUNDS THEREFOR</h4>
<p>Approved on February 12, 2025</p>
</center>
<p>Be it enacted by the Senate and House of Representatives of the Philippines in Congress assembled:</p>
<p>Section 1. Short Title. - This Act shall be known as the 'E-Governance Act'.</p>
<p>Section 2. Declaration of Policy. - It is hereby declared the policy of the State to promote digital transformation across all branches of government.</p>
<p>Section 3. Definition of Terms. - For purposes of this Act, ICT shall mean Information and Communications Technology.</p>
</body>
</html>
"""

SAMPLE_CASE_HTML = """
<html>
<head><title>G.R. No. 233922</title></head>
<body>
<center>
<h3>THIRD DIVISION</h3>
<p><b>G.R. No. 233922, October 16, 2024</b></p>
<p><b>PEOPLE OF THE PHILIPPINES, PETITIONER, vs. JUAN DELA CRUZ, RESPONDENT.</b></p>
<p><b>CAGUIOA, <i>J.</i>:</b></p>
</center>
<p>Before this Court is a Petition for Review on Certiorari under Rule 45 of the Rules of Court...</p>
<p>The essential elements of the crime must be established beyond reasonable doubt.</p>
</body>
</html>
"""

def test_clean_html_to_text():
    service = LegalIngestionService()
    cleaned = service.clean_html_to_text(SAMPLE_LAWS_HTML)
    
    assert "The Lawphil Project" not in cleaned["cleaned_text"]
    assert "REPUBLIC ACT NO. 12254" in cleaned["cleaned_text"]
    assert "E-Governance Act" in cleaned["cleaned_text"]
    assert cleaned["word_count"] > 20

def test_extract_legal_metadata_statute():
    service = LegalIngestionService()
    cleaned = service.clean_html_to_text(SAMPLE_LAWS_HTML)
    meta = service.extract_legal_metadata(cleaned["cleaned_text"], html_title="Republic Act No. 12254", url="https://www.lawphil.net/statutes/repacts/ra2025/ra_12254_2025.html")
    
    assert meta["category"] == "Republic Act"
    assert meta["doc_number"] == "RA 12254"
    assert meta["year"] == 2025
    assert "12254" in meta["title"]
    assert "E-GOVERNANCE" in meta["title"]

def test_extract_legal_metadata_jurisprudence():
    service = LegalIngestionService()
    cleaned = service.clean_html_to_text(SAMPLE_CASE_HTML)
    meta = service.extract_legal_metadata(cleaned["cleaned_text"], html_title="G.R. No. 233922", url="https://www.lawphil.net/judjuris/juri2024/oct2024/gr_233922_2024.html")
    
    assert meta["category"] == "Jurisprudence"
    assert "G.R. No. 233922" in meta["doc_number"]
    assert meta["year"] == 2024
    assert meta["ponente"] == "CAGUIOA"
    assert "PEOPLE OF THE PHILIPPINES vs. JUAN DELA CRUZ" in meta["title"]

def test_chunk_document():
    service = LegalIngestionService()
    text = "Section 1. Short Title. " * 50
    chunks = service.chunk_document(text)
    
    assert len(chunks) >= 1
    for chk in chunks:
        assert len(chk) >= 50

def test_preview_from_raw():
    service = LegalIngestionService()
    preview = service.preview_from_raw(SAMPLE_LAWS_HTML, is_html=True)
    
    assert preview["chunk_count"] >= 1
    assert preview["metadata"]["category"] == "Republic Act"
    assert preview["metadata"]["year"] == 2025
    assert len(preview["sample_chunks"]) >= 1

@patch("requests.get")
def test_fetch_url(mock_get):
    mock_resp = MagicMock()
    mock_resp.text = SAMPLE_LAWS_HTML
    mock_resp.status_code = 200
    mock_resp.encoding = "utf-8"
    mock_get.return_value = mock_resp
    
    service = LegalIngestionService()
    html = service.fetch_url("https://www.lawphil.net/statutes/repacts/ra2025/ra_12254_2025.html")
    assert "REPUBLIC ACT NO. 12254" in html
