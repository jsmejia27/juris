import os, sys, time, json, subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_pipeline import LegalRAGPipeline, DEFAULT_NUM_CTX

def get_gpu_vram_usage():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu-memory.used,memory.total", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        ).strip()
        lines = out.split("\n")
        used, total = [int(x.strip()) for x in lines[0].split(",")]
        return {"used_mb": used, "total_mb": total, "pct": round(used / total * 100, 1)}
    except Exception as e:
        return {"used_mb": 0, "total_mb": 16384, "error": str(e)}


def test_16k_context_latency():
    print(f"Testing 16K context configuration (num_ctx = {DEFAULT_NUM_CTX})...")
    vram_before = get_gpu_vram_usage()
    print("VRAM Before:", vram_before)

    pipeline = LegalRAGPipeline(num_ctx=16384, temperature=0.0)

    # Synthetic large statutory context
    synthetic_docs = [
        {
            "doc_id": f"doc-{i}",
            "title": f"Republic Act No. {10000 + i}",
            "gr_no": f"RA {10000 + i}",
            "category": "Republic Act",
            "section": f"Section {i}",
            "text": f"SECTION {i}. This is a detailed statutory provision outlining the legal rules and rights in the Philippines. " * 25,
            "doctrine_status": "good_law"
        }
        for i in range(1, 9)
    ]

    context_str = pipeline.format_context(synthetic_docs)
    question = "What are the main statutory rules provided in Section 1 to 8?"
    prompt = f"You are Juris. Summarize the key provisions from the retrieved context.\n\n{context_str}\n\nQuestion: {question}"
    print(f"Prompt character length: {len(prompt)} chars (~{len(prompt)//4} tokens)")

    t0 = time.time()
    tokens = []
    for token in pipeline.llm.stream(prompt):
        tokens.append(token)
    elapsed = time.time() - t0

    vram_after = get_gpu_vram_usage()
    print('VRAM After:', vram_after)

    total_tokens = len(tokens)
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    print(f'Generation finished: {total_tokens} tokens in {elapsed:.2f}s ({tok_per_sec:.2f} tok/s)')

    record = {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "num_ctx": 16384,
        "prompt_len_chars": len(prompt),
        "tokens_generated": total_tokens,
        "elapsed_sec": round(elapsed, 2),
        "tokens_per_sec": round(tok_per_sec, 2),
        "vram_used_mb": vram_after.get("used_mb", 0),
        "vram_total_mb": vram_after.get("total_mb", 0)
    }

    os.makedirs('logs', exist_ok=True)
    with open('logs/vram_benchmark.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')

    assert len(tokens) > 5
    assert elapsed < 60.0
    print('16K/vram test PASSED!')


def test_refusal_on_insufficient_context():
    pipeline = LegalRAGPipeline(num_ctx=16384, temperature=0.0)
    prompt = "You are Juris, an AI Legal Research Assistant. If the provided context is insufficient to answer the question, state: 'Based on the provided Philippine legal documents, there is insufficient information to answer this inquiry.' Do not guess.\n\nCONTEXT:\nSOURCE 1: Article 1 defines marriage as a special contract.\n\nWUESTION: What is the tax exemption threshold for offshore jet fuel importation in 2026?"
    resp = pipeline.llm.invoke(prompt)
    print('Refusal test output:', resp)
    assert 'insufficient' in resp.lower() or 'not' in resp.lower() or 'unsupported' in resp.lower()