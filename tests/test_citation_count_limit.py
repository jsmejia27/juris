import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from server import ChatRequest, pipeline
from rag_pipeline import deduplicate_sources

def test_chat_request_default_top_k():
    req = ChatRequest(message="What is cyber libel?")
    assert req.top_k == 4

def test_retriever_limit_4():
    docs = pipeline.retriever.retrieve("What is cyber libel?", limit=4)
    assert len(docs) <= 4

def test_dedup_capped_at_4():
    sample_sources = [
        {"title": f"Doc {i}", "gr_no": f"G.R. No. {100000 + i}", "category": "Jurisprudence", "score": 0.9 - (i * 0.05)}
        for i in range(10)
    ]
    deduped = deduplicate_sources(sample_sources)[:4]
    assert len(deduped) == 4
