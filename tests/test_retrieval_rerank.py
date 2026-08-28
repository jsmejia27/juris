import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from rag_pipeline import extract_lexical_anchors, apply_lexical_anchor_boost

def test_extract_lexical_anchors():
    q1 = 'What are the penalties under RA 9262 Section 5?'
    anchors = extract_lexical_anchors(q1)
    assert 'ra 9262' in anchors or '9262' in anchors
    assert any('5' in a or 'sec' in a or 'section' in a for a in anchors)

    q2 = 'What was held in G.R. No. 196359 (Tan-Andal)?'
    anchors2 = extract_lexical_anchors(q2)
    assert any('196359' in a for a in anchors2)

def test_lexical_anchor_boost():
    query = 'What is Republic Act No. 11861 Section 8?'
    candidates = [
        {
            'doc_id': 'doc-1',
            'title': 'General Family Law',
            'gr_no': 'RA 8972',
            'text': 'Parental provisions for solo parents in general.',
            'score': 0.85
        },
        {
            'doc_id': 'doc-2',
            'title': 'Expanded Solo Parents Welfare Act',
            'gr_no': 'RA 11861',
            'text': 'Section 8. Educational Benefits and Scholarships.',
            'score': 0.70
        }
    ]
    boosted = apply_lexical_anchor_boost(candidates, query)
    assert boosted[0]['doc_id'] == 'doc-2'
    assert boosted[0]['score'] > boosted[1]['score']
    assert boosted[0]['boost_tier'] == 'tier_ra_with_section'

def test_tiered_anchor_boost_specificity():
    query = "Under G.R. No. 196359 and Section 5 of RA 9262"
    candidates = [
        {
            'doc_id': 'doc-gr',
            'title': 'Tan-Andal',
            'gr_no': 'G.R. No. 196359',
            'text': 'Psychological incapacity landmark decision.',
            'score': 0.01
        },
        {
            'doc_id': 'doc-ra-sec',
            'title': 'VAWC',
            'gr_no': 'RA 9262',
            'text': 'Section 5 acts of violence against women.',
            'score': 0.01
        },
        {
            'doc_id': 'doc-bare-sec',
            'title': 'Unrelated Act',
            'gr_no': 'RA 7610',
            'text': 'Section 5 special protection of children.',
            'score': 0.01
        }
    ]
    boosted = apply_lexical_anchor_boost(candidates, query)
    assert boosted[0]['doc_id'] == 'doc-gr'
    assert boosted[0]['boost_tier'] == 'tier_gr_docket'
    assert boosted[0]['lexical_boost'] == 0.30

    assert boosted[1]['doc_id'] == 'doc-ra-sec'
    assert boosted[1]['boost_tier'] == 'tier_ra_with_section'
    assert boosted[1]['lexical_boost'] == 0.25

    assert boosted[2]['doc_id'] == 'doc-bare-sec'
    assert boosted[2]['boost_tier'] == 'tier_bare_section'
    assert boosted[2]['lexical_boost'] == 0.05

def test_dual_reranker_execution():
    from rag_pipeline import LegalCrossEncoderRanker
    candidates = [
        {"category": "Republic Act", "title": "RA 9262", "gr_no": "RA 9262", "text": "Section 5 violence against women"},
        {"category": "Republic Act", "title": "RA 7610", "gr_no": "RA 7610", "text": "Section 10 child abuse"}
    ]
    # Test FlashRank
    r_tiny = LegalCrossEncoderRanker(model_name="ms-marco-TinyBERT-L-2-v2")
    res_tiny = r_tiny.rerank_passages("violence against women penalties", candidates, top_k=2)
    assert len(res_tiny) == 2
    assert "reranker_model" in res_tiny[0]

    # Test Cross-Encoder fallback / mock
    with patch("fastembed.rerank.cross_encoder.TextCrossEncoder") as mock_ce:
        mock_instance = MagicMock()
        mock_instance.rerank.return_value = [0.95, 0.40]
        mock_ce.return_value = mock_instance
        
        r_bge = LegalCrossEncoderRanker(model_name="bge-reranker-base")
        res_bge = r_bge.rerank_passages("violence against women penalties", candidates, top_k=2)
        assert len(res_bge) == 2

def test_resolve_model_execution_path():
    from rag_pipeline import resolve_model_execution_path

    # Direct lookup stays local
    res_local = resolve_model_execution_path("What is Section 5 of RA 9262?", enable_frontier=True)
    assert res_local["execution_path"] == "local"
    assert res_local["complexity"] == "DIRECT_LOOKUP"

    # Multi-doctrine synthesis routes to frontier when enabled
    res_frontier = resolve_model_execution_path("Is Tan-Andal controlling over Molina and how does it apply?", enable_frontier=True)
    assert res_frontier["execution_path"] == "frontier"
    assert res_frontier["complexity"] == "MULTI_DOCTRINE_SYNTHESIS"

    # When frontier disabled, falls back to local
    res_fallback = resolve_model_execution_path("Is Tan-Andal controlling over Molina?", enable_frontier=False)
    assert res_fallback["execution_path"] == "local"

