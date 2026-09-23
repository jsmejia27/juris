import sys
sys.path.insert(0, ".")
from server import pipeline

queries = [
    "accidental injury between friends can they sue for damages",
    "tort liability accidental injury civil suit negligence",
    "can my friend sue me if they get injured accidentally in the philippines"
]

for q in queries:
    print("=" * 60)
    print("QUERY:", q)
    docs = pipeline.retriever.retrieve(q, limit=7)
    print(f"Retrieved {len(docs)} documents:")
    for i, d in enumerate(docs, 1):
        print(f"{i}. [{d.get('category')}] {d.get('title')} | Citation: {d.get('gr_no')} | Section: {d.get('section')} | Score: {d.get('score'):.4f}")
