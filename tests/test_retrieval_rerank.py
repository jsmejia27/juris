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
