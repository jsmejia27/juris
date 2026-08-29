import os
import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from server import app, verify_admin
from legal_ingestion_service import LegalIngestionService

client = TestClient(app)

def test_check_existing_document_not_found(tmp_path, monkeypatch):
    log_file = str(tmp_path / "test_ingestion_log.jsonl")
    monkeypatch.setattr("legal_ingestion_service.INGESTION_LOG_FILE", log_file)
    
    with patch("legal_ingestion_service.QdrantClient") as mock_qdrant:
        mock_instance = mock_qdrant.return_value
        mock_instance.scroll.return_value = ([], None)
        
        service = LegalIngestionService()
        res = service.check_existing_document(doc_number="RA 99999", title="Non-existent Act", source_url="https://lawphil.net/test.html")
        assert res["is_duplicate"] is False
        assert res["match_count"] == 0

def test_check_existing_document_found_in_log(tmp_path, monkeypatch):
    log_file = str(tmp_path / "test_ingestion_log.jsonl")
    monkeypatch.setattr("legal_ingestion_service.INGESTION_LOG_FILE", log_file)
    
    sample_entry = {
        "doc_id": "test-doc-123",
        "title": "Republic Act No. 12254",
        "category": "Republic Act",
        "doc_number": "RA 12254",
        "year": 2025,
        "source_url": "https://www.lawphil.net/statutes/repacts/ra2025/ra_12254_2025.html",
        "chunks_count": 8,
        "ingested_at": "2026-08-28 12:00:00 UTC"
    }
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_entry) + "\n")

    with patch("legal_ingestion_service.QdrantClient"):
        service = LegalIngestionService()
        res = service.check_existing_document(doc_number="RA 12254")
        assert res["is_duplicate"] is True
        assert res["matched_by"] == "doc_number"
        assert len(res["existing_records"]) == 1
        assert res["existing_records"][0]["doc_id"] == "test-doc-123"

def test_scan_all_duplicates(tmp_path, monkeypatch):
    log_file = str(tmp_path / "test_ingestion_log.jsonl")
    monkeypatch.setattr("legal_ingestion_service.INGESTION_LOG_FILE", log_file)

    entries = [
        {"doc_id": "d1", "doc_number": "RA 12254", "source_url": "https://lawphil.net/1.html", "title": "RA 12254 Ver 1"},
        {"doc_id": "d2", "doc_number": "RA 12254", "source_url": "https://lawphil.net/1.html", "title": "RA 12254 Ver 2"},
        {"doc_id": "d3", "doc_number": "RA 8972", "source_url": "https://lawphil.net/2.html", "title": "RA 8972 Solo Parent"}
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    with patch("legal_ingestion_service.QdrantClient"):
        service = LegalIngestionService()
        clusters = service.scan_all_duplicates()
        assert len(clusters) >= 1
        assert any("RA 12254" in c["cluster_key"] for c in clusters)

def test_delete_document_from_qdrant(tmp_path, monkeypatch):
    log_file = str(tmp_path / "test_ingestion_log.jsonl")
    monkeypatch.setattr("legal_ingestion_service.INGESTION_LOG_FILE", log_file)

    entries = [
        {"doc_id": "d1", "doc_number": "RA 12254", "source_url": "https://lawphil.net/1.html", "title": "RA 12254"},
        {"doc_id": "d2", "doc_number": "RA 8972", "source_url": "https://lawphil.net/2.html", "title": "RA 8972"}
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    with patch("legal_ingestion_service.QdrantClient") as mock_qdrant:
        mock_instance = mock_qdrant.return_value
        service = LegalIngestionService()
        res = service.delete_document_from_qdrant(doc_number="RA 12254")
        assert res["status"] == "success"
        mock_instance.delete.assert_called_once()

        # Check log file updated
        history = service.get_ingestion_history()
        assert len(history) == 1
        assert history[0]["doc_number"] == "RA 8972"

def test_api_check_duplicate_endpoint():
    app.dependency_overrides[verify_admin] = lambda: "admin"
    try:
        with patch("legal_ingestion_service.LegalIngestionService.check_existing_document") as mock_chk:
            mock_chk.return_value = {"is_duplicate": True, "match_count": 1, "existing_records": [{"title": "RA 12254"}]}
            
            resp = client.post(
                "/api/manage/ingest/check-duplicate",
                json={"doc_number": "RA 12254", "title": "", "source_url": ""}
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_duplicate"] is True
    finally:
        app.dependency_overrides.pop(verify_admin, None)

def test_api_scan_duplicates_endpoint():
    app.dependency_overrides[verify_admin] = lambda: "admin"
    try:
        with patch("legal_ingestion_service.LegalIngestionService.scan_all_duplicates") as mock_scan:
            mock_scan.return_value = [{"cluster_key": "RA 12254", "duplicate_count": 2}]
            
            resp = client.get("/api/manage/ingest/duplicates")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_clusters"] == 1
    finally:
        app.dependency_overrides.pop(verify_admin, None)

def test_api_delete_endpoint():
    app.dependency_overrides[verify_admin] = lambda: "admin"
    try:
        with patch("legal_ingestion_service.LegalIngestionService.delete_document_from_qdrant") as mock_del:
            mock_del.return_value = {"status": "success", "message": "Purged"}
            
            resp = client.post(
                "/api/manage/ingest/delete",
                json={"doc_number": "RA 12254"}
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "success"
    finally:
        app.dependency_overrides.pop(verify_admin, None)
