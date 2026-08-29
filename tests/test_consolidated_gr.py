import sys, re
sys.path.insert(0, ".")
from rag_pipeline import extract_lexical_anchors_with_tiers

queries = [
    "What was held in G.R. Nos. 162335 & 162605?",
    "G.R. Nos. 162335, 162605",
    "G.R. No. 162335 and 162605",
    "G.R. No. 162335 and G.R. No. 162605"
]

for q in queries:
    anchors = extract_lexical_anchors_with_tiers(q)
    print(f"Query: {q}")
    for a in anchors:
        print(f"  Tier: {a['tier']}, Raw: {a['raw']}, Terms: {a['terms']}")
    print("-" * 50)
