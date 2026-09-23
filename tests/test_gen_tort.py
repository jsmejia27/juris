import sys, re
sys.path.insert(0, ".")
from server import pipeline
from rag_pipeline import PROMPT_TAB1_TREATISE

LEGAL_CONCEPT_EXPANSIONS = {
    r"\b(?:accidental(?:ly)?\s+injur\w*|sue\s+(?:me\s+)?for\s+injur\w*|friend\s+injur\w*|injur\w*\s+friend|sports\s+injur\w*|slip\s+and\s+fall|negligen\w*|vehicular\s+accident|car\s+crash|hit\s+and\s+run|injured\s+accidentally)\b": 
        "quasi-delict culpa aquiliana Article 2176 Civil Code negligence damages proximate cause assumption of risk fortuitous event",
}

def expand_query(q):
    for pat, legal_terms in LEGAL_CONCEPT_EXPANSIONS.items():
        if re.search(pat, q, re.IGNORECASE):
            return f"{q} {legal_terms}"
    return q

query = "can my friend sue me if they get injured accidentally in the philippines"
exp_query = expand_query(query)
docs = pipeline.retriever.retrieve(exp_query, limit=7)
ctx = pipeline.format_context(docs)
prompt = PROMPT_TAB1_TREATISE.format(history_section="", context=ctx, question=query)

print("Streaming generation...")
chunks = []
for c in pipeline.llm.stream(prompt):
    chunks.append(c)

full_text = "".join(chunks)
print(f"Total length: {len(full_text)} chars")
print("=" * 60)
print(full_text[:800])
print("...")
print(full_text[-600:])
