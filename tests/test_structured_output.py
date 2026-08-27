import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pytest
from verifier import (
    Claim,
    StructuredLegalResponse,
    parse_and_validate_structured_output,
    verify_citations_and_claims
)


def test_parse_valid_json():
    raw_json = json.dumps({
        "claims": [
            {
                "text": "Replic Act No. 9262 penalizes acts of violence against women and children.",
                "source_id": "RA 9262",
                "quoted_text": "Violence against women and their children refers to any act or a series of acts...",
                "source_type": "statute"
            }
        ],
        "answer_prose": "Under **Replic Act No. 9262**, acts of violence against women and children are strictly penalized. [RA 9262]"
    })
    result = parse_and_validate_structured_output(raw_json)
    assert isinstance(result, StructuredLegalResponse)
    assert len(result.claims) == 1
    assert result.claims[0].source_id == "RA 9262"


def test_parse_markdown_wrapped_json():
    raw = '''```json
{
  "claims": [
    {
      "text": "Tan-Andal ruling",
      "source_id": "G.R. No. 196359",
      "quoted_text": "Psychological incapacity",
      "source_type": "jurisprudence"
    }
  ],
  "answer_prose": "In Tan-Andal..."
}
```'''
    result = parse_and_validate_structured_output(raw)
    assert len(result.claims) == 1
    assert result.claims[0].source_id == "G.R. No. 196359"


def test_parse_fallback():
    malformed = "Under RA 9262, the penalties apply. [RA 9262]"
    result = parse_and_validate_structured_output(malformed)
    assert isinstance(result, StructuredLegalResponse)
    assert len(result.claims) >= 1


def test_verify_valid():
    retrieved = [{
        'doc_id': 'ra-9262',
        'gr_no': 'RA 9262',
        'title': 'vawc',
        'text': 'Section 5. Acts of violence against women and children...'
    }]
    resp = StructuredLegalResponse(
        claims=[Claim(
            text='Claim 1',
            source_id='RA 9262',
            quoted_text='Acts of violence against women',
            source_type='statute'
        )],
        answer_prose='Prose [RA 9262]'
    )
    summary = verify_citations_and_claims(resp, retrieved)
    assert summary.verified_claims == 1
    assert summary.accuracy_rate == 1.0


def test_verify_hallucinated():
    retrieved = [{
        'doc_id': 'ra-8972',
        'gr_no': 'RA 8972',
        'title': 'Solo Parents',
        'text': 'Parental leave of 7 days...'
    }]
    resp = StructuredLegalResponse(
        claims=[Claim(
            text='Bad claim',
            source_id='RA 99999',
           quoted_text='Fake quote',
            source_type='statute'
        )],
        answer_prose='Prose [RA 99999]'
    )
    summary = verify_citations_and_claims(resp, retrieved)
    assert summary.verified_claims == 0
    assert summary.unverified_claims == 1
    assert len(summary.failures) == 1
