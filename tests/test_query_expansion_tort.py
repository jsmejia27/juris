import sys, re
sys.path.insert(0, ".")
from server import pipeline

LEGAL_CONCEPT_EXPANSIONS = {
    r"\b(?:accidental(?:ly)?\s+injur\w*|sue\s+(?:me\s+)?for\s+injur\w*|friend\s+injur\w*|injur\w*\s+friend|sports\s+injur\w*|slip\s+and\s+fall|negligen\w*|vehicular\s+accident|car\s+crash|hit\s+and\s+run|injured\s+accidentally)\b": 
        "quasi-delict culpa aquiliana Article 2176 Civil Code negligence damages proximate cause assumption of risk fortuitous event",
}

def expand_query(q):
    expanded = q
    for pat, legal_terms in LEGAL_CONCEPT_EXPANSIONS.items():
        if re.search(pat, q, re.IGNORECASE):
            expanded = f"{q} {legal_terms}"
            break
    return expanded

query = "can my friend sue me if they get injured accidentally in the philippines"
exp_query = expand_query(query)
print("Expanded query:", exp_query)

docs = pipeline.retriever.retrieve(exp_query, limit=7)
print(f"Retrieved {len(docs)} documents:")
for i, d in enumerate(docs, 1):
    print(f"{i}. [{d.get('category')}] {d.get('title')} | Citation: {d.get('gr_no')} | Score: {d.get('score'):.4f}")
