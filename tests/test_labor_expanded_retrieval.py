import sys, re
sys.path.insert(0, ".")
from server import pipeline

query = "i was absent at work for 2 days, i was not able to go to work because i was sick and was not able to inform work. when i came, i learned i was fired."

LABOR_TAXONOMY = (
    "Labor Code Presidential Decree 442 Article 297 Article 282 just cause illegal dismissal "
    "gross and habitual neglect of duty abandonment of work procedural due process twin-notice rule "
    "two notice requirement separation pay backwages NLRC reinstatement disease illness Article 299 Article 284"
)

exp_query = f"{query} {LABOR_TAXONOMY}"
docs = pipeline.retriever.retrieve(exp_query, limit=5)
print(f"Retrieved {len(docs)} documents with expanded labor taxonomy:")
for i, d in enumerate(docs, 1):
    print(f"  {i}. [{d.get('category')}] {d.get('title')} ({d.get('gr_no')}) - score: {d.get('score', 0):.4f}")
