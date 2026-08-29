import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient
from server import app
from legal_ingestion_service import LegalIngestionService

client = TestClient(app)

def test_get_document_record_unauthorized():
    res = client.post("/api/manage/ingest/record", json={"doc_number": "RA 386"})
    assert res.status_code == 401

def test_get_document_record_service():
    service = LegalIngestionService()
    # Test checking Civil Code or any indexed statute in Qdrant
    res = service.get_full_document_record(doc_number="RA 386")
    if res.get("found"):
        assert "metadata" in res
        assert "full_text" in res
        assert "chunks" in res
        assert len(res["chunks"]) > 0
        assert res["metadata"]["category"] is not None

def test_get_document_record_api_authenticated():
    from server import create_session_token
    token = create_session_token("admin")
    client.cookies.set("juris_admin_token", token)
    res = client.post(
        "/api/manage/ingest/record",
        json={"doc_number": "RA 386"}
    )
    assert res.status_code == 200
    data = res.json()
    if data.get("found"):
        assert "metadata" in data
        assert "full_text" in data
        assert "chunks" in data
