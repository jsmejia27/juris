import os, sys, time, json, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_pipeline import LegalRAGPipeline, LegalRetriever
from verifier import parse_and_validate_structured_output, verify_citations_and_claims
from router import route_query

def run_benchmark_evaluation():
    print("*** Starting Philippine Legal RAG Eval Harness **j")
    with open("eval/benchmark_queries.json", "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    pipeline = LegalRAGPipeline(num_ctx=16384, temperature=0.0)
    eval_results = []
    total_verification_scores = []
    total_latencies = []

    for item in benchmark_data:
        qid = item["id"]
        query = item["query"]
        cat = item["category"]
        should_refuse = item["should_refuse"]

        print(f"\n[EVAL] Running {qid} (category: {cat}): {query}")

        t0 = time.time()
        routing = route_query(query)
        retrieved = pipeline.retriever.retrieve(query, limit=8)

        context_str = pipeline.format_context(retrieved)
        prompt = f"You are Juris, an AI Legal Research Assistant.\n\nCONTEXT:\n{context_str}\n\nQUESTION: {query}"
        raw = pipeline.llm.invoke(prompt)
        elapsed = time.time() - t0

        structured = parse_and_validate_structured_output(raw)
        v_summary = verify_citations_and_claims(structured, retrieved, query=query)

        refused = "insufficient" in raw.lower() or "not provided" in raw.lower()
        refusal_correct = (bool(should_refuse) == refused)

        total_verification_scores.append(v_summary.accuracy_rate)
        total_latencies.append(elapsed)

        res_record = {
            "id": qid,
            "query": query,
            "complexity": routing["complexity"],
            "retrieved_count": len(retrieved),
            "verified_claims": v_summary.verified_claims,
            "unverified_claims": v_summary.unverified_claims,
            "citation_accuracy": v_summary.accuracy_rate,
            "refusal_correct": refusal_correct,
            "latency_sec": round(elapsed, 2)
        }
        eval_results.append(res_record)
        print(f"  -> Accuracy: {v_summary.accuracy_rate*100:.1f}%, Latency: {elapsed:.2f}s")

    mean_acc = sum(total_verification_scores) / len(total_verification_scores)
    mean_lat = sum(total_latencies) / len(total_latencies)

    summary_report = {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "total_queries": len(benchmark_data),
        "mean_citation_accuracy": round(mean_acc, 3),
        "mean_latency_sec": round(mean_lat, 2),
        "query_details": eval_results
    }

    os.makedirs("eval", exist_ok=True)
    with open("eval/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print("\n====================================================")
    print("BENCHMARK EVALUATION COMPLETED")
    print(f"Mean Citation Accuracy: {mean_acc*100:.1f}%")
    print(f"Mean Latency: {mean_lat:.2f}s")
    print("====================================================\n")

    assert mean_acc >= 0.85

if __name__ == "__main__":
    run_benchmark_evaluation()