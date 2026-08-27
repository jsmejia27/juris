# server.py
import os
import json
import asyncio
import threading
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, BackgroundTasks, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pyarrow.parquet as pq
import secrets

from rag_pipeline import (
    LegalRAGPipeline,
    get_shared_qdrant_client,
    _QDRANT_LOCK,
    SYSTEM_PROMPT_TEMPLATE,
    DEFAULT_STORAGE_DIR,
    DEFAULT_COLLECTION,
    deduplicate_sources,
    resolve_model_execution_path
)
from verifier import parse_and_validate_structured_output, verify_citations_and_claims
from router import route_query
from ingest_data import (
    process_repacts_records,
    process_juris_records,
    process_consolidated_records,
    split_legal_sections,
    load_checkpoint,
    save_checkpoint,
    init_qdrant_collection
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http import models
import uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field, field_validator

ALLOWED_ORIGINS = [
    origin.strip() for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:9010,http://127.0.0.1:9010,http://0.0.0.0:9010"
    ).split(",") if origin.strip()
]

ALLOWED_CHAT_MODELS = {
    "qwen3.5:9b",
    "qwen3:14b",
    "llama3.2:latest",
    "phi4-mini-reasoning:latest",
    "qwen3.8:latest",
    "qwen2.5vl:latest"
}

app = FastAPI(title="Juris - Philippine Legal AI Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize pipeline as singleton with Qwen 3.5 9B on RTX 5070 Ti
pipeline = LegalRAGPipeline(llm_model="qwen3.5:9b", temperature=0.1, num_ctx=8192)

# Ingestion state tracking
INGESTION_STATE = {
    "status": "idle", # "idle", "running", "stopping", "completed", "error"
    "source": None,
    "current_doc": 0,
    "total_docs": 0,
    "current_chunk": 0,
    "total_chunks": 0,
    "progress_percent": 0.0,
    "logs": [],
    "should_stop": False,
    "error_message": None
}

class ChatMessage(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant|system)$")
    content: str = Field(..., max_length=10000)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: Optional[List[ChatMessage]] = Field(default=[], max_length=20)
    model: Optional[str] = Field(default="qwen3.5:9b", max_length=64)
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=1.0)
    num_ctx: Optional[int] = Field(default=8192, ge=1024, le=16384)
    category: Optional[str] = Field(default="All", max_length=50)
    top_k: Optional[int] = Field(default=4, ge=1, le=16)
    year_min: Optional[int] = Field(default=1901, ge=1900, le=2026)
    year_max: Optional[int] = Field(default=2026, ge=1900, le=2026)

    @field_validator("model")
    def sanitize_model(cls, v):
        if v not in ALLOWED_CHAT_MODELS:
            return "qwen3.5:9b"
        return v

class IngestJobRequest(BaseModel):
    source: str = "repacts" # "repacts", "juris", "all"
    limit: Optional[int] = 500
    batch_size: Optional[int] = 128

@app.get("/api/models")
async def get_models():
    return {
        "models": [
            "qwen3.5:9b",
            "qwen3:14b",
            "llama3.2:latest",
            "phi4-mini-reasoning:latest",
            "qwen3.8:latest",
            "qwen2.5vl:latest"
        ],
        "default": pipeline.llm_model,
        "gpu_accelerator": "NVIDIA GeForce RTX 5070 Ti (16GB VRAM)"
    }

@app.get("/api/admin/stats")
async def legacy_admin_stats():
    # Graceful alias for older browser sessions
    return {"status": "ok", "notice": "Please visit /manage-juris for authenticated management"}

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    pipeline.set_model(req.model)
    pipeline.set_temperature(req.temperature)
    pipeline.set_num_ctx(req.num_ctx)

    async def event_generator():
        # 0. Route Query and Resolve Model Execution Path
        route_decision = resolve_model_execution_path(req.message, requested_model=req.model)
        logger.info(f"Query routing decision: path={route_decision['execution_path']}, complexity={route_decision['complexity']}")
        yield f"data: {json.dumps({'type': 'routing', 'decision': route_decision})}\n\n"

        # Contextualize Follow-up Query for Retrieval
        history_dicts = [{"role": h.role, "content": h.content} for h in (req.history or [])]
        retrieval_query = pipeline.contextualize_query(req.message, history_dicts)

        # 1. Retrieve Local Statutes & Jurisprudence (Qdrant Two-Stage Retrieval)
        category_filter = None if req.category == "All" else req.category
        loop = asyncio.get_event_loop()
        sources = await loop.run_in_executor(
            None,
            lambda: pipeline.retriever.retrieve(
                query=retrieval_query,
                limit=req.top_k,
                category=category_filter,
                year_min=req.year_min,
                year_max=req.year_max
            )
        )

        # 2. Query Pending Legislative Bills from Open Congress API (BetterGov.ph)
        all_sources = list(sources)
        bills_context = ""
        try:
            bills_data = await loop.run_in_executor(
                None,
                lambda: pipeline.congress_client.search_bills(retrieval_query, limit_per_chamber=2)
            )
            bills_context = pipeline.congress_client.format_bills_context(bills_data)

            for sb in bills_data.get("senate_bills", []):
                all_sources.append({
                    "category": "Senate Bill",
                    "title": sb["title"],
                    "gr_no": sb["bill_name"],
                    "date": sb.get("date_filed", ""),
                    "score": 0.99,
                    "source_url": sb.get("url", ""),
                    "ponente": sb.get("authors", ""),
                    "doc_id": sb.get("id", sb["bill_name"])
                })
            for hb in bills_data.get("house_bills", []):
                all_sources.append({
                    "category": "House Bill",
                    "title": hb["title"],
                    "gr_no": hb["bill_name"],
                    "date": hb.get("date_filed", ""),
                    "score": 0.99,
                    "source_url": hb.get("url", ""),
                    "ponente": hb.get("authors", ""),
                    "doc_id": hb.get("id", hb["bill_name"])
                })
        except Exception as err:
            logger.warning(f"Non-blocking Open Congress query exception: {err}")

        # Deduplicate all statutory sources and bills so each authority appears only once
        deduped_sources = deduplicate_sources(all_sources)

        # Send combined sources event
        yield f"data: {json.dumps({'type': 'sources', 'sources': deduped_sources})}\n\n"
        await asyncio.sleep(0.01)

        # 3. Format Context & Multi-Turn Prompt
        context_str = pipeline.format_context(sources, bills_context=bills_context)
        history_section = pipeline.format_history_section(history_dicts)
        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            history_section=history_section,
            context=context_str,
            question=req.message
        )

        # Stream LLM tokens asynchronously
        accumulated_response = []
        for chunk in pipeline.llm.stream(prompt):
            accumulated_response.append(chunk)
            payload = json.dumps({"type": "token", "token": chunk})
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.002)

        # 4. Perform citation and claim verification pass
        try:
            full_text = "".join(accumulated_response)
            structured_resp = parse_and_validate_structured_output(full_text)
            v_summary = verify_citations_and_claims(structured_resp, sources, query=req.message)
            yield f"data: {json.dumps({'type': 'verification', 'summary': v_summary.dict()})}\n\n"
        except Exception as v_err:
            logger.debug(f"Verification pass error: {v_err}")

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ==========================================
# AUTHENTICATION & JURIS MANAGEMENT ENDPOINTS
# ==========================================

security = HTTPBasic()

ADMIN_USERNAME = os.getenv("JURIS_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("JURIS_ADMIN_PASSWORD", "Mangaldan2026")

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials for Juris Management Console.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/api/manage/stats")
async def get_admin_stats(auth: str = Depends(verify_admin)):
    # 1. Parquet files info
    datasets = []
    parquet_files = [
        {"file": "repacts_with_summary.parquet", "label": "Republic Acts (with Summaries)", "type": "repacts"},
        {"file": "repacts.parquet", "label": "Republic Acts (Raw)", "type": "repacts_raw"},
        {"file": "juris.parquet", "label": "Supreme Court Jurisprudence", "type": "juris"},
        {"file": "consolidated.parquet", "label": "Consolidated Legal Corpus", "type": "consolidated"}
    ]

    total_available_docs = 0
    for item in parquet_files:
        fname = item["file"]
        if os.path.exists(fname):
            pf = pq.ParquetFile(fname)
            num_rows = pf.metadata.num_rows
            size_kb = os.path.getsize(fname) / 1024
            datasets.append({
                "file": fname,
                "label": item["label"],
                "type": item["type"],
                "total_rows": num_rows,
                "size_kb": size_kb
            })

    # 2. Ingested Checkpoint info
    ckpt_path = os.path.join(DEFAULT_STORAGE_DIR, "ingested_docs.json")
    ingested_set = load_checkpoint(DEFAULT_STORAGE_DIR)
    total_ingested = len(ingested_set)

    repacts_ingested = len([i for i in ingested_set if "repacts" in i.lower()])
    juris_ingested = len([i for i in ingested_set if "juris" in i.lower()])

    # True available unique valid documents (12,000 Republic Acts + 66,759 Valid SC Decisions)
    total_available_docs = 78759

    # 3. Storage size
    storage_size_kb = 0
    if os.path.exists(DEFAULT_STORAGE_DIR):
        for dirpath, dirnames, filenames in os.walk(DEFAULT_STORAGE_DIR):
            for f in filenames:
                try:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        storage_size_kb += os.path.getsize(fp) / 1024
                except OSError:
                    pass

    # 4. Total points in Qdrant collection
    total_vector_points = 0
    try:
        client = get_shared_qdrant_client()
        if client.collection_exists(DEFAULT_COLLECTION):
            col_info = client.get_collection(DEFAULT_COLLECTION)
            total_vector_points = col_info.points_count or 0
    except Exception as e:
        logger.warning(f"Could not fetch collection points: {e}")

    overall_pct = min(100.0, (total_ingested / total_available_docs * 100)) if total_available_docs > 0 else 0

    return {
        "datasets": datasets,
        "total_available_docs": total_available_docs,
        "total_ingested_docs": total_ingested,
        "repacts_ingested": repacts_ingested,
        "juris_ingested": juris_ingested,
        "total_vector_points": total_vector_points,
        "storage_size_kb": storage_size_kb,
        "overall_percentage": round(overall_pct, 2),
        "gpu": "NVIDIA GeForce RTX 5070 Ti (16GB VRAM)",
        "ingestion_state": INGESTION_STATE
    }

def run_background_ingestion(source: str, limit: Optional[int], batch_size: int):
    global INGESTION_STATE
    INGESTION_STATE["status"] = "running"
    INGESTION_STATE["source"] = source
    INGESTION_STATE["should_stop"] = False
    INGESTION_STATE["error_message"] = None
    INGESTION_STATE["progress_percent"] = 0.0
    INGESTION_STATE["logs"] = []

    def log_msg(msg: str):
        logger.info(msg)
        INGESTION_STATE["logs"].append(msg)
        if len(INGESTION_STATE["logs"]) > 100:
            INGESTION_STATE["logs"].pop(0)

    try:
        log_msg(f"🚀 Starting background ingestion for source='{source}', limit={limit}, batch_size={batch_size}")
        client = get_shared_qdrant_client()
        init_qdrant_collection(client, DEFAULT_COLLECTION, vector_size=768)

        ingested_ids = load_checkpoint(DEFAULT_STORAGE_DIR)
        log_msg(f"Loaded existing checkpoint: {len(ingested_ids)} documents already indexed.")

        records = []
        if source == "repacts":
            repacts_path = "repacts_with_summary.parquet" if os.path.exists("repacts_with_summary.parquet") else "repacts.parquet"
            records.extend(process_repacts_records(repacts_path, limit=limit, exclude_ids=ingested_ids))

        elif source == "juris":
            juris_path = "juris.parquet"
            records.extend(process_juris_records(juris_path, limit=limit, exclude_ids=ingested_ids))

        elif source in ["consolidated", "all"]:
            if os.path.exists("consolidated.parquet"):
                records.extend(process_consolidated_records("consolidated.parquet", limit=limit, exclude_ids=ingested_ids))
            else:
                repacts_path = "repacts_with_summary.parquet" if os.path.exists("repacts_with_summary.parquet") else "repacts.parquet"
                records.extend(process_repacts_records(repacts_path, limit=limit, exclude_ids=ingested_ids))
                juris_path = "juris.parquet"
                records.extend(process_juris_records(juris_path, limit=limit, exclude_ids=ingested_ids))

        INGESTION_STATE["total_docs"] = len(records)
        log_msg(f"Loaded {len(records)} candidate documents. Starting text chunking...")

        all_chunks_payload = []
        all_chunks_text = []

        for rec in records:
            if rec["doc_id"] in ingested_ids:
                continue
            chunks = split_legal_sections(rec["content"], rec["category"])
            total_chunks = len(chunks)
            for idx, chunk in enumerate(chunks):
                contextualized_text = f"[{rec['category']}] {rec['title']} ({rec['gr_no']}): {chunk}"
                all_chunks_text.append(contextualized_text)
                all_chunks_payload.append({
                    "doc_id": rec["doc_id"],
                    "source": rec["source"],
                    "category": rec["category"],
                    "title": rec["title"],
                    "gr_no": rec["gr_no"],
                    "date": rec["date"],
                    "year": rec["year"],
                    "ponente": rec["ponente"],
                    "summary": rec["summary"],
                    "key_provisions": rec["key_provisions"],
                    "tags": rec["tags"],
                    "keywords": rec["keywords"],
                    "chunk_index": idx + 1,
                    "total_chunks": total_chunks,
                    "text": chunk
                })

        total_chunks = len(all_chunks_text)
        INGESTION_STATE["total_chunks"] = total_chunks
        log_msg(f"Generated {total_chunks} vector chunks to index.")

        if total_chunks == 0:
            log_msg("All available documents in this selection are already indexed in Qdrant.")
            INGESTION_STATE["status"] = "idle"
            INGESTION_STATE["progress_percent"] = 100.0
            return

        points_buffer = []
        dense_embedder = pipeline.retriever.dense_embedder
        sparse_embedder = pipeline.retriever.sparse_embedder

        for i in range(0, total_chunks, batch_size):
            if INGESTION_STATE["should_stop"]:
                log_msg("🛑 Ingestion cancelled by user request.")
                INGESTION_STATE["status"] = "idle"
                save_checkpoint(DEFAULT_STORAGE_DIR, ingested_ids)
                return

            batch_texts = all_chunks_text[i:i + batch_size]
            batch_payloads = all_chunks_payload[i:i + batch_size]

            dense_vectors = dense_embedder.embed_documents(batch_texts)
            sparse_vectors = list(sparse_embedder.embed(batch_texts))

            for text, payload, d_vec, s_vec in zip(batch_texts, batch_payloads, dense_vectors, sparse_vectors):
                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": d_vec,
                        "sparse": models.SparseVector(
                            indices=s_vec.indices.tolist(),
                            values=s_vec.values.tolist()
                        )
                    },
                    payload=payload
                )
                points_buffer.append(point)

            with _QDRANT_LOCK:
                client.upsert(collection_name=DEFAULT_COLLECTION, points=points_buffer)
            points_buffer.clear()

            batch_doc_ids = {p["doc_id"] for p in batch_payloads}
            ingested_ids.update(batch_doc_ids)

            INGESTION_STATE["current_chunk"] = min(i + batch_size, total_chunks)
            INGESTION_STATE["progress_percent"] = round((INGESTION_STATE["current_chunk"] / total_chunks) * 100, 1)

            if (i // batch_size) % 5 == 0:
                save_checkpoint(DEFAULT_STORAGE_DIR, ingested_ids)
                log_msg(f"Indexed {INGESTION_STATE['current_chunk']}/{total_chunks} chunks ({INGESTION_STATE['progress_percent']}%)")

        save_checkpoint(DEFAULT_STORAGE_DIR, ingested_ids)
        INGESTION_STATE["status"] = "completed"
        INGESTION_STATE["progress_percent"] = 100.0
        log_msg(f"✅ Ingestion job completed successfully! {total_chunks} chunks stored.")

    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        INGESTION_STATE["status"] = "error"
        INGESTION_STATE["error_message"] = str(e)
        log_msg(f"❌ Error during ingestion: {str(e)}")

@app.post("/api/manage/ingest/start")
async def start_ingestion(job: IngestJobRequest, auth: str = Depends(verify_admin)):
    global INGESTION_STATE
    if INGESTION_STATE["status"] == "running":
        return JSONResponse(status_code=400, content={"error": "An ingestion job is already running."})

    thread = threading.Thread(
        target=run_background_ingestion,
        args=(job.source, job.limit, job.batch_size),
        daemon=True
    )
    thread.start()
    return {"message": "Ingestion job started", "job": job.dict()}

@app.post("/api/manage/ingest/stop")
async def stop_ingestion(auth: str = Depends(verify_admin)):
    global INGESTION_STATE
    if INGESTION_STATE["status"] == "running":
        INGESTION_STATE["should_stop"] = True
        return {"message": "Stop signal sent to ingestion job"}
    return {"message": "No active ingestion job to stop"}

# Mount Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_landing():
    with open("static/landing.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/chat", response_class=HTMLResponse)
async def read_chat():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/disclaimer", response_class=HTMLResponse)
async def read_disclaimer():
    with open("static/disclaimer.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/manage-juris", response_class=HTMLResponse)
async def read_manage_juris(auth: str = Depends(verify_admin)):
    with open("static/admin.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=9010, reload=False)
