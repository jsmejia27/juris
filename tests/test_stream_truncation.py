import sys
sys.path.insert(0, ".")
import time
from server import pipeline
from rag_pipeline import PROMPT_TAB1_TREATISE

print("Retrieving docs...")
docs = pipeline.retriever.retrieve('prescriptive period of cyber libel', limit=7)
print(f"Retrieved {len(docs)} documents.")

ctx = pipeline.format_context(docs)
print(f"Context length (chars): {len(ctx)}")

prompt = PROMPT_TAB1_TREATISE.format(history_section="", context=ctx, question="what is the prescriptive period of cyber libel?")
print(f"Full prompt length (chars): {len(prompt)}")

print("Streaming generation from LLM...")
start = time.time()
chunks = []
for c in pipeline.llm.stream(prompt):
    chunks.append(c)

elapsed = time.time() - start
full_text = "".join(chunks)
print(f"Generation completed in {elapsed:.2f}s.")
print(f"Chunks count: {len(chunks)}, Text length: {len(full_text)} chars")
print("="*40)
print("FIRST 200 CHARS:")
print(full_text[:200])
print("="*40)
print("LAST 400 CHARS:")
print(full_text[-400:])
print("="*40)
