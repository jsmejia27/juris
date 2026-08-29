# tests/test_admin_ingest_api.py
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from server import app, create_session_token, ADMIN_USERNAME

client = TestClient(app)

SAMPLE_LAWS_HTML = """
<html>
<head><title>Republic Act No. 12254</title></head>
<body>
<h3>REPUBLIC ACT NO. 12254</h3>
<h4>AN ACT INSTITUTIONALIZING THE TRANSITION OF THE GOVERNMENT TO E-GOVERNANCE, STRENGTHENING THE ICT ACADEMY, AND APPROPRIATING FUNDS THEREFOR</h4>
<p>Approved on February 12, 2025</p>
<p>Section 1. Short Title. - This Act shall be known as the 'E-Governance Act'.</p>
<p>Section 2. Declaration of Policy. - It is hereby declared the policy of the State to promote digital transformation across all branches of government.</p>
</body>
</html>
"""

def test_admin_ingest_unauthorized():
    # Without cookie
    client.cookies.clear()
    res = client.post("/api/manage/ingest/preview-url", json={"url": "https://www.lawphil.net/test.html"})
    assert res.status_code == 401

def test_admin_ingest_preview_raw_authenticated():
    token = create_session_token(ADMIN_USERNAME)
    client.cookies.set("juris_admin_token", token)
    
    res = client.post(
        "/api/manage/ingest/preview-raw",
        json={
            "content": SAMPLE_LAWS_HTML,
            "is_html": True,
            "title": "RA 12254 Test"
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert "metadata" in data
    assert data["metadata"]["category"] == "Republic Act"
    assert data["chunk_count"] >= 1
    assert len(data["sample_chunks"]) >= 1

@patch("requests.get")
def test_admin_ingest_preview_url_authenticated(mock_get):
    mock_resp = MagicMock()
    mock_resp.text = SAMPLE_LAWS_HTML
    mock_resp.status_code = 200
    mock_resp.encoding = "utf-8"
    mock_get.return_value = mock_resp

    token = create_session_token(ADMIN_USERNAME)
    client.cookies.set("juris_admin_token", token)

    res = client.post(
        "/api/manage/ingest/preview-url",
        json={"url": "https://www.lawphil.net/statutes/repacts/ra2025/ra_12254_2025.html"}
    )
    assert res.status_code == 200
    data = res.json()
    assert "metadata" in data
    assert "E-GOVERNANCE" in data["metadata"]["title"]
    assert data["metadata"]["year"] == 2025

def test_admin_ingest_history_authenticated():
    token = create_session_token(ADMIN_USERNAME)
    client.cookies.set("juris_admin_token", token)

    res = client.get("/api/manage/ingest/history")
    assert res.status_code == 200
    data = res.json()
    assert "history" in data
