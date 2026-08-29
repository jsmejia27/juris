# tests/test_query_digest_stream.py
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from langchain_ollama import OllamaLLM
from server import app

client = TestClient(app)

@patch.object(OllamaLLM, "stream")
def test_stream_executive_digest(mock_stream):
    mock_stream.return_value = ["### Executive Summary\n\n", "This is a test digest."]

    res = client.post(
        "/api/query/digest/stream",
        json={
            "question": "What is the Solo Parent Leave benefit?",
            "treatise": "Republic Act No. 8972 provides solo parents with 7 days leave.",
            "sources": []
        }
    )
    assert res.status_code == 200
    content = res.text
    assert "data: " in content
    assert "tab2_done" in content
    assert "done" in content
