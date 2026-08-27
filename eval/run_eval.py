import os
import sys
import time
import json
import logging
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_pipeline import LegalRAGPipeline, LegalRetriever, resolve_model_execution_path, apply_lexical_anchor_boost, LegalCrossEncoderRanker
from verifier import parse_and_validate_structured_output, verify_citations_and_claims
from router import route_query

logger = logging.getLogger(__name__)

BENCHMARK_PATH = "eval/benchmark_queries.json"
EVAL_RESULTS_PATH = "eval/eval_results.json"
RERANKER_AB_PATH = "eval/reranker_ab_results.json"

def check_source_match(expected_list: List[str], doc: Dict[str, Any]) -> bool:
    if not expected_list:
        return False
    doc_text = (
        str(doc.get("gr_no") or "") + " " +
        str(doc.get("title") or "") + " " +
        str(doc.get("doc_id") or "") + " " +
        str(doc.get("text") or "")
    ).lower()

    for exp in expected_list:
        exp_clean = exp.lower().strip()
        if exp_clean in doc_text:
            return True
    return False

def run_reranker_ab_evaluation(sample_limit: int = 15):
    """
    A/B Evaluation Harness comparing 'ms-marco-TinyBERT-L-2-v2' against 'BAAI/bge-reranker-base'
    on the exact same top-50 candidate sets.
    """
    print("\n========================================================")
    print("🚀 RUNNING RERANKER A/B COMPARISON: TinyBERT vs BGE-Base")
    print("========================================================")

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)[:sample_limit]

    retriever = LegalRetriever()
    ranker_tiny = LegalCrossEncoderRanker(model_name="ms-marco-TinyBERT-L-2-v2")
    ranker_bge = LegalCrossEncoderRanker(model_name="bge-reranker-base")

    tiny_hits = 0
    bge_hits = 0
    pool_hits = 0
    valid_queries = 0

    tiny_latencies = []
    bge_latencies = []
    comparisons = []

    for item in benchmark_data:
        qid = item["id"]
        query = item["query"]
        expected = item.get("expected_sources", [])
        if item.get("should_refuse") or not expected:
            continue

        valid_queries += 1

        # 1. Fetch top-50 candidate pool
        q_dense = retriever.dense_embedder.embed_query(query)
        q_sparse = list(retriever.sparse_embedder.embed([query]))[0]

        from qdrant_client.http import models
        from rag_pipeline import _QDRANT_LOCK
        with _QDRANT_LOCK:
            raw_pts = retriever.client.query_points(
                collection_name=retriever.collection_name,
                prefetch=[
                    models.Prefetch(query=q_dense, using="dense", limit=50),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=q_sparse.indices.tolist(),
                            values=q_sparse.values.tolist()
                        ),
                        using="sparse",
                        limit=50
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=50
            )

        candidates = []
        for p in raw_pts.points:
            payload = p.payload or {}
            candidates.append({
                "score": p.score,
                "doc_id": payload.get("doc_id", ""),
                "title": payload.get("title", ""),
                "gr_no": payload.get("gr_no", ""),
                "category": payload.get("category", ""),
                "text": payload.get("text", "")
            })

        boosted = apply_lexical_anchor_boost(candidates, query)

        # Check if expected source in top 50
        in_pool = any(check_source_match(expected, doc) for doc in boosted[:50])
        if in_pool: pool_hits += 1

        # Rerank with TinyBERT
        t0 = time.time()
        top_tiny = ranker_tiny.rerank_passages(query, boosted[:50], top_k=8)
        lat_tiny = time.time() - t0
        tiny_latencies.append(lat_tiny)
        tiny_hit = any(check_source_match(expected, doc) for doc in top_tiny)
        if tiny_hit: tiny_hits += 1

        # Rerank with BGE-Base
        t1 = time.time()
        top_bge = ranker_bge.rerank_passages(query, boosted[:50], top_k=8)
        lat_bge = time.time() - t1
        bge_latencies.append(lat_bge)
        bge_hit = any(check_source_match(expected, doc) for doc in top_bge)
        if bge_hit: bge_hits += 1

        comparisons.append({
            "id": qid,
            "query": query,
            "expected": expected,
            "in_top50_pool": in_pool,
            "tinybert_top8_hit": tiny_hit,
            "bge_top8_hit": bge_hit,
            "tinybert_latency_ms": round(lat_tiny * 1000, 1),
            "bge_latency_ms": round(lat_bge * 1000, 1)
        })

    recall_pool = (pool_hits / valid_queries) if valid_queries else 1.0
    recall_tiny = (tiny_hits / valid_queries) if valid_queries else 1.0
    recall_bge = (bge_hits / valid_queries) if valid_queries else 1.0

    avg_lat_tiny = (sum(tiny_latencies) / len(tiny_latencies) * 1000) if tiny_latencies else 0.0
    avg_lat_bge = (sum(bge_latencies) / len(bge_latencies) * 1000) if bge_latencies else 0.0

    ab_summary = {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "evaluated_queries": valid_queries,
        "candidate_pool_recall_at_50": round(recall_pool, 3),
        "tinybert": {
            "model": "ms-marco-TinyBERT-L-2-v2",
            "recall_at_8": round(recall_tiny, 3),
            "avg_latency_ms": round(avg_lat_tiny, 1)
        },
        "bge_reranker_base": {
            "model": "BAAI/bge-reranker-base",
            "recall_at_8": round(recall_bge, 3),
            "avg_latency_ms": round(avg_lat_bge, 1)
        },
        "recall_delta_bge_vs_tiny": round(recall_bge - recall_tiny, 3),
        "comparisons": comparisons
    }

    with open(RERANKER_AB_PATH, "w", encoding="utf-8") as f:
        json.dump(ab_summary, f, indent=2)

    print(f"Candidate Pool Recall@50 : {recall_pool*100:.1f}%")
    print(f"TinyBERT Recall@8        : {recall_tiny*100:.1f}% ({avg_lat_tiny:.1f}ms)")
    print(f"BGE-Reranker-Base Recall@8: {recall_bge*100:.1f}% ({avg_lat_bge:.1f}ms)")
    print(f"Recall Delta (BGE - Tiny): {ab_summary['recall_delta_bge_vs_tiny']*100:+.1f}%")
    print("========================================================\n")
    return ab_summary

def run_full_benchmark_evaluation(sample_limit: Optional[int] = None):
    print("\n========================================================")
    print("🚀 STARTING JURIS END-TO-END PIPELINE BENCHMARK EVALUATION")
    print("========================================================")

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    if sample_limit:
        benchmark_data = benchmark_data[:sample_limit]

    pipeline = LegalRAGPipeline(num_ctx=16384, temperature=0.0)

    query_details = []
    total_accuracies = []
    total_latencies = []
    total_tokens_per_sec = []

    failure_counts = {"source_not_retrieved": 0, "quote_mismatch": 0}
    routing_counts = {"local": 0, "frontier": 0}
    recall_8_hits = 0
    valid_retrieval_queries = 0

    for idx, item in enumerate(benchmark_data):
        qid = item["id"]
        query = item["query"]
        cat = item["category"]
        expected_sources = item.get("expected_sources", [])
        should_refuse = item.get("should_refuse", False)

        print(f"[{idx+1}/{len(benchmark_data)}] Evaluating {qid} ({cat}): {query[:60]}...")

        t_start = time.time()

        # 1. Routing decision
        route_info = resolve_model_execution_path(query)
        exec_path = route_info["execution_path"]
        routing_counts[exec_path] = routing_counts.get(exec_path, 0) + 1

        # 2. Retrieval & Reranking
        retrieved_top8 = pipeline.retriever.retrieve(query, limit=8)

        # Measure Recall@8
        if expected_sources and not should_refuse:
            valid_retrieval_queries += 1
            hit_8 = any(check_source_match(expected_sources, doc) for doc in retrieved_top8)
            if hit_8:
                recall_8_hits += 1

        # 3. Context & Prompt Assembly
        context_str = pipeline.format_context(retrieved_top8)
        prompt = (
            f"You are Juris, an AI Legal Research Assistant.\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION: {query}"
        )

        # 4. Generation
        gen_start = time.time()
        raw_output = pipeline.llm.invoke(prompt)
        gen_elapsed = time.time() - gen_start
        total_elapsed = time.time() - t_start

        # Calculate approximate token throughput
        approx_tokens = len(raw_output) / 4.0
        tok_sec = approx_tokens / gen_elapsed if gen_elapsed > 0 else 0.0

        # 5. Verification & Failure Analysis
        structured = parse_and_validate_structured_output(raw_output)
        v_summary = verify_citations_and_claims(structured, retrieved_top8, query=query)

        # Count specific failure modes
        for fail in v_summary.failures:
            ftype = fail.get("failure_type", "source_not_retrieved")
            failure_counts[ftype] = failure_counts.get(ftype, 0) + 1

        # Refusal check
        refused = "insufficient" in raw_output.lower() or "not provided" in raw_output.lower()
        refusal_accurate = (bool(should_refuse) == refused)

        total_accuracies.append(v_summary.accuracy_rate)
        total_latencies.append(total_elapsed)
        total_tokens_per_sec.append(tok_sec)

        unsupported_rate = (v_summary.unverified_claims / v_summary.total_claims) if v_summary.total_claims > 0 else 0.0

        rec = {
            "id": qid,
            "query": query,
            "category": cat,
            "execution_path": exec_path,
            "complexity": route_info["complexity"],
            "total_claims": v_summary.total_claims,
            "verified_claims": v_summary.verified_claims,
            "unverified_claims": v_summary.unverified_claims,
            "citation_accuracy": round(v_summary.accuracy_rate, 3),
            "unsupported_claim_rate": round(unsupported_rate, 3),
            "refusal_accurate": refusal_accurate,
            "latency_sec": round(total_elapsed, 2),
            "tokens_per_sec": round(tok_sec, 1),
            "failures": v_summary.failures
        }
        query_details.append(rec)
        print(f"   -> Accuracy: {v_summary.accuracy_rate*100:.1f}%, Unverified: {v_summary.unverified_claims}, Latency: {total_elapsed:.2f}s ({tok_sec:.1f} tok/s)")

    mean_acc = sum(total_accuracies) / len(total_accuracies) if total_accuracies else 0.0
    mean_lat = sum(total_latencies) / len(total_latencies) if total_latencies else 0.0
    mean_tok_sec = sum(total_tokens_per_sec) / len(total_tokens_per_sec) if total_tokens_per_sec else 0.0
    recall_8_rate = (recall_8_hits / valid_retrieval_queries) if valid_retrieval_queries else 1.0

    report = {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "total_queries_evaluated": len(benchmark_data),
        "overall_citation_accuracy": round(mean_acc, 3),
        "overall_retrieval_recall_at_8": round(recall_8_rate, 3),
        "mean_latency_sec": round(mean_lat, 2),
        "mean_throughput_tok_sec": round(mean_tok_sec, 1),
        "failure_type_breakdown": failure_counts,
        "routing_distribution": routing_counts,
        "query_details": query_details
    }

    os.makedirs("eval", exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n========================================================")
    print("📈 FINAL BENCHMARK EVALUATION METRICS REPORT")
    print("========================================================")
    print(f"Overall Citation Accuracy       : {mean_acc*100:.1f}%")
    print(f"Retrieval Recall@8              : {recall_8_rate*100:.1f}%")
    print(f"Mean Latency Per Request        : {mean_lat:.2f}s")
    print(f"Mean Generation Throughput      : {mean_tok_sec:.1f} tokens/sec")
    print(f"Failure Breakdown (Not Retrieved): {failure_counts.get('source_not_retrieved', 0)}")
    print(f"Failure Breakdown (Quote Mismatch): {failure_counts.get('quote_mismatch', 0)}")
    print(f"Routing Distribution (Local/Frontier): {routing_counts}")
    print("========================================================\n")

    return report

if __name__ == "__main__":
    run_reranker_ab_evaluation(sample_limit=15)
    run_full_benchmark_evaluation(sample_limit=5)