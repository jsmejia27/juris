# tests/test_server_endpoints.py
import pytest
import os
import sys
import io
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server import app

@pytest.fixture
def client(tmp_path, monkeypatch):
    test_vault_dir = str(tmp_path / "server_vault_cases")
    monkeypatch.setattr("case_vault_service.VAULT_BASE_DIR", test_vault_dir)
    return TestClient(app)

def test_drafting_templates_api(client):
    res = client.get("/api/drafting/templates")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert len(data["templates"]) >= 5

def test_export_docx_api(client):
    payload = {
        "draft_text": "# REPUBLIC OF THE PHILIPPINES\n\nMOTION FOR RECONSIDERATION\n\nDiscussion points here.",
        "title": "MR_Appeal"
    }
    res = client.post("/api/drafting/export-docx", json=payload)
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert len(res.content) > 1000

def test_case_vault_api_lifecycle(client):
    # 1. Create Case
    create_payload = {
        "title": "Abad v. Gomez Realty Corp",
        "docket_number": "CA-G.R. SP No. 182910",
        "court": "Court of Appeals (Special 4th Division)",
        "case_type": "Civil",
        "parties": "Spouses Abad vs. Gomez Realty Corp.",
        "description": "Petition for Certiorari under Rule 65"
    }
    res = client.post("/api/vault/cases", json=create_payload)
    assert res.status_code == 200
    case_data = res.json()["case"]
    case_id = case_data["case_id"]

    # 2. Get Case Details
    res = client.get(f"/api/vault/cases/{case_id}")
    assert res.status_code == 200
    assert res.json()["case"]["title"] == "Abad v. Gomez Realty Corp"

    # 3. Upload Document to Case
    file_content = b"COMPLAINT FOR SPECIFIC PERFORMANCE AND DAMAGES. Plaintiff purchased Lot 4 Block 2 on installment basis."
    files = {
        "file": ("Complaint.txt", io.BytesIO(file_content), "text/plain")
    }
    data = {"doc_category": "Pleading / Complaint"}
    res = client.post(f"/api/vault/cases/{case_id}/upload", files=files, data=data)
    assert res.status_code == 200
    assert res.json()["document"]["filename"] == "Complaint.txt"

    # 4. Query Case Vault
    res = client.post(f"/api/vault/cases/{case_id}/query", json={"query": "installment purchase Lot 4", "top_k": 3})
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    # 5. Delete Case
    res = client.delete(f"/api/vault/cases/{case_id}")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
