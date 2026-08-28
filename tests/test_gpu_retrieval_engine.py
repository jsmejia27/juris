# tests/test_gpu_retrieval_engine.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from legal_retrieval_engine import LegalModelManager, LegalRetrievalService, RetrievedChunk

def test_retrieved_chunk_model():
    chunk = RetrievedChunk(
        id="chunk-101",
        doc_id="ra_11861",
        title="Expanded Solo Parents Welfare Act",
        category="Republic Act",
        law_no="RA 11861",
        text="Section 4. Rights and privileges of solo parents.",
        rrf_score=0.95,
        rerank_score=0.88
    )
    assert chunk.id == "chunk-101"
    assert chunk.law_no == "RA 11861"
    assert chunk.rerank_score == 0.88
    assert chunk.doctrine_status == "good_law"

def test_model_manager_encoding_format():
    manager = LegalModelManager.get_instance()
    dense, indices, values = manager.encode_query_bge_m3("Solo Parents Act RA 11861")
    assert isinstance(dense, list)
    assert len(dense) in (768, 1024)
    assert isinstance(indices, list)
    assert isinstance(values, list)
    assert len(indices) == len(values)

def test_model_manager_rerank_scoring():
    manager = LegalModelManager.get_instance()
    query = "Solo parent leaves benefits"
    passages = [
        "Republic Act 11861 grants 7 days parental leave to solo parents.",
        "The Fisheries Code regulates municipal waters and licensing."
    ]
    scores = manager.rerank_pairs(query, passages)
    assert len(scores) == 2
    # The first passage should score significantly higher than the unrelated one
    assert scores[0] > scores[1]

@pytest.mark.anyio
async def test_retrieval_service_mocked():
    service = LegalRetrievalService()
    
    # Mock Qdrant query_points response
    mock_point_1 = MagicMock()
    mock_point_1.id = "p-1"
    mock_point_1.score = 0.95
    mock_point_1.payload = {
        "doc_id": "ra_11861",
        "title": "Expanded Solo Parents Act",
        "category": "Republic Act",
        "law_no": "RA 11861",
        "text": "Parental leave of seven days shall be granted to solo parents."
    }
    
    mock_point_2 = MagicMock()
    mock_point_2.id = "p-2"
    mock_point_2.score = 0.80
    mock_point_2.payload = {
        "doc_id": "gr_196359",
        "title": "Tan-Andal vs. Andal",
        "category": "Jurisprudence",
        "gr_no": "G.R. No. 196359",
        "text": "Psychological incapacity is not a medical illness."
    }

    mock_resp = MagicMock()
    mock_resp.points = [mock_point_1, mock_point_2]
    
    service.client.query_points = AsyncMock(return_value=mock_resp)

    results = await service.retrieve_and_rerank_legal_context(
        query="solo parent leave benefits",
        top_k_candidates=2,
        final_top_k=2,
        score_threshold=0.0
    )

    assert len(results) == 2
    assert results[0].law_no == "RA 11861"
    assert results[0].rerank_score >= results[1].rerank_score

def test_api_legal_search_endpoint():
    from fastapi.testclient import TestClient
    from server import app
    client = TestClient(app)
    
    response = client.post(
        "/api/legal/search",
        json={"query": "solo parent leave", "top_k": 2, "score_threshold": 0.0}
    )
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert "total_retrieved" in data
