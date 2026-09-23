import sys
sys.path.insert(0, ".")
from server import pipeline
from rag_pipeline import PROMPT_TAB1_TREATISE, deduplicate_sources

query = "i was absent at work for 2 days, i was not able to go to work because i was sick and was not able to inform work. when i came, i learned i was fired."

sources = pipeline.retriever.retrieve(query, limit=5)
deduped = deduplicate_sources(sources)[:6]
ctx = pipeline.format_context(deduped)
prompt = PROMPT_TAB1_TREATISE.format(history_section="", context=ctx, question=query)

print("Streaming full treatise with 24K context and 8K predict limit...")
chunks = []
for c in pipeline.llm.stream(prompt):
    chunks.append(c)

full_text = "".join(chunks)
print(f"Total response length: {len(full_text)} characters (~{len(full_text.split())} words)")
print("Has legal_planning closing tag:", "</legal_planning>" in full_text)
print("=" * 60)
print(full_text[:800])
print("...")
print(full_text[-600:])
