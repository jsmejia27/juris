import sys, os, time
sys.path.insert(0, ".")
from server import pipeline
from rag_pipeline import expand_legal_taxonomy_query, PROMPT_TAB1_TREATISE

query = "i was absent at work for 2 days, i was not able to go to work because i was sick and was not able to inform work. when i came, i learned i was fired."
exp_query = expand_legal_taxonomy_query(query)
print(f"Original Query: {query}")
print(f"Expanded Query: {exp_query}")

docs = pipeline.retriever.retrieve(query, limit=5)
print(f"\nRetrieved {len(docs)} documents:")
for i, d in enumerate(docs, 1):
    print(f"  {i}. [{d.get('category')}] {d.get('title')} ({d.get('gr_no')}) - score: {d.get('score', 0):.4f}")
