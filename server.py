import os
import json
import asyncio
import threading
import logging
import time
import hmac
import hashlib
import struct
import base64
import urllib.parse
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, Response, BackgroundTasks, Depends, HTTPException, status, Cookie, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import pyarrow.parquet as pq
import secrets

from rag_pipeline import (
    LegalRAGPipeline,
    get_shared_qdrant_client,
    _QDRANT_LOCK,
    SYSTEM_PROMPT_TEMPLATE,
    PROMPT_TAB1_TREATISE,
    PROMPT_TAB2_EDITORIAL,
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

# Initialize pipeline as singleton with Qwen 3.5 9B on RTX 5070 Ti (16K context)
pipeline = LegalRAGPipeline(llm_model="qwen3.5:9b", temperature=0.1, num_ctx=16384)

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
    num_ctx: Optional[int] = Field(default=16384, ge=1024, le=32768)
    category: Optional[str] = Field(default="All", max_length=50)
    top_k: Optional[int] = Field(default=5, ge=1, le=16)
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
        effective_limit = min(req.top_k or 5, 5)
        sources = await loop.run_in_executor(
            None,
            lambda: pipeline.retriever.retrieve(
                query=retrieval_query,
                limit=effective_limit,
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
                lambda: pipeline.congress_client.search_bills(retrieval_query, limit_per_chamber=1)
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

        # Deduplicate all statutory sources and bills, capping to 5-6 citations only
        deduped_sources = deduplicate_sources(all_sources)[:6]

        # Send combined sources event
        yield f"data: {json.dumps({'type': 'sources', 'sources': deduped_sources})}\n\n"
        await asyncio.sleep(0.01)

        # 3. Format Context & Multi-Turn Prompts for Dual Tabs
        context_str = pipeline.format_context(deduped_sources, bills_context=bills_context)
        history_section = pipeline.format_history_section(history_dicts)
        
        prompt_tab1 = PROMPT_TAB1_TREATISE.format(
            history_section=history_section,
            context=context_str,
            question=req.message
        )
        prompt_tab2 = PROMPT_TAB2_EDITORIAL.format(
            history_section=history_section,
            context=context_str,
            question=req.message
        )

        # Stream Tab 1: In-Depth Legal Treatise
        accumulated_tab1 = []
        for chunk in pipeline.llm.stream(prompt_tab1):
            accumulated_tab1.append(chunk)
            payload = json.dumps({"type": "token", "tab": 1, "token": chunk})
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.002)

        yield f"data: {json.dumps({'type': 'tab1_done'})}\n\n"

        # 4. Perform citation and claim verification pass on Tab 1
        if accumulated_tab1:
            try:
                full_text = "".join(accumulated_tab1).strip()
                if full_text:
                    structured_resp = parse_and_validate_structured_output(full_text)
                    if structured_resp.answer_prose:
                        v_summary = verify_citations_and_claims(structured_resp, sources, query=req.message)
                        yield f"data: {json.dumps({'type': 'verification', 'tab': 1, 'summary': v_summary.dict()})}\n\n"
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

class DigestStreamRequest(BaseModel):
    question: str = Field(..., min_length=2)
    treatise: Optional[str] = ""
    sources: Optional[List[Dict[str, Any]]] = []
    history: Optional[List[ChatMessage]] = []
    model: Optional[str] = None

@app.post("/api/query/digest/stream")
async def stream_executive_digest(req: DigestStreamRequest):
    async def digest_event_generator():
        try:
            history_dicts = [{"role": h.role, "content": h.content} for h in (req.history or [])]
            history_section = pipeline.format_history_section(history_dicts)
            
            context_str = pipeline.format_context(req.sources or [])
            if not context_str.strip() and req.treatise:
                context_str = f"PREVIOUSLY SYNTHESIZED LEGAL TREATISE:\n{req.treatise}"

            prompt_tab2 = PROMPT_TAB2_EDITORIAL.format(
                history_section=history_section,
                context=context_str,
                question=req.question
            )

            for chunk in pipeline.llm.stream(prompt_tab2):
                payload = json.dumps({"type": "token", "tab": 2, "token": chunk})
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.002)

            yield f"data: {json.dumps({'type': 'tab2_done'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logger.error(f"Error streaming executive digest: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        digest_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ==========================================
# 2-FACTOR AUTHENTICATION (TOTP) & SESSION MANAGEMENT
# ==========================================

ADMIN_USERNAME = os.getenv("JURIS_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("JURIS_ADMIN_PASSWORD", "jurisadmin")
SESSION_SECRET = os.getenv("JURIS_SESSION_SECRET", "juris_secure_session_secret_2025")

OTP_CONFIG_PATH = os.path.join("config", "admin_otp.json")

def get_or_create_otp_secret() -> str:
    env_secret = os.getenv("JURIS_ADMIN_OTP_SECRET")
    if env_secret and len(env_secret.strip()) >= 16:
        return env_secret.strip().upper()
    
    os.makedirs("config", exist_ok=True)
    if os.path.exists(OTP_CONFIG_PATH):
        try:
            with open(OTP_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "secret" in data and len(data["secret"]) >= 16:
                    return data["secret"].strip().upper()
        except Exception as e:
            logger.warning(f"Error reading {OTP_CONFIG_PATH}: {e}")

    # Generate a fresh 32-character base32 secret (20 bytes)
    raw = secrets.token_bytes(20)
    secret = base64.b32encode(raw).decode("utf-8").replace("=", "").upper()
    try:
        with open(OTP_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "secret": secret,
                "created_at": time.time(),
                "issuer": "Juris Philippine Legal AI",
                "user": ADMIN_USERNAME
            }, f, indent=2)
        logger.info(f"Created new Admin OTP Secret in {OTP_CONFIG_PATH}")
    except Exception as e:
        logger.warning(f"Could not persist {OTP_CONFIG_PATH}: {e}")
    return secret

def generate_totp(secret_str: str, time_val: Optional[int] = None) -> str:
    if time_val is None:
        time_val = int(time.time())
    clean_secret = secret_str.strip().upper()
    padding = "=" * ((8 - len(clean_secret) % 8) % 8)
    key = base64.b32decode((clean_secret + padding), casefold=True)
    counter = int(time_val // 30)
    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    binary_code = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(binary_code % 1000000).zfill(6)

def verify_totp(otp_input: str, secret_str: str, window: int = 1) -> bool:
    if not otp_input:
        return False
    clean_otp = str(otp_input).strip()
    if len(clean_otp) != 6 or not clean_otp.isdigit():
        return False
    now = int(time.time())
    for w in range(-window, window + 1):
        expected = generate_totp(secret_str, now + (w * 30))
        if secrets.compare_digest(expected, clean_otp):
            return True
    return False

def create_session_token(username: str) -> str:
    timestamp = str(int(time.time()))
    payload = f"{username}:{timestamp}"
    signature = hmac.new(SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{signature}"

def verify_session_token(token: str, max_age_seconds: int = 28800) -> Optional[str]:
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return None
        username, timestamp_str, sig = parts
        timestamp = int(timestamp_str)
        if time.time() - timestamp > max_age_seconds:
            return None
        expected_sig = hmac.new(SESSION_SECRET.encode(), f"{username}:{timestamp_str}".encode(), hashlib.sha256).hexdigest()
        if secrets.compare_digest(sig, expected_sig):
            return username
        return None
    except Exception:
        return None

async def verify_admin(
    request: Request,
    juris_admin_token: Optional[str] = Cookie(default=None),
    authorization: Optional[str] = Header(default=None),
    x_admin_otp: Optional[str] = Header(default=None)
) -> str:
    # 1. Check Cookie Session Token
    if juris_admin_token:
        user = verify_session_token(juris_admin_token)
        if user and secrets.compare_digest(user, ADMIN_USERNAME):
            return user

    # 2. Check Bearer Token
    if authorization and authorization.startswith("Bearer "):
        bearer_token = authorization[7:].strip()
        user = verify_session_token(bearer_token)
        if user and secrets.compare_digest(user, ADMIN_USERNAME):
            return user

    # 3. Check HTTP Basic Auth + OTP Header or Password:OTP format
    if authorization and authorization.startswith("Basic "):
        try:
            encoded_creds = authorization[6:].strip()
            decoded = base64.b64decode(encoded_creds).decode("utf-8")
            if ":" in decoded:
                u, p = decoded.split(":", 1)
                secret = get_or_create_otp_secret()
                otp_candidate = x_admin_otp
                if not otp_candidate and ":" in p:
                    p, otp_candidate = p.rsplit(":", 1)

                correct_user = secrets.compare_digest(u, ADMIN_USERNAME)
                correct_pass = secrets.compare_digest(p, ADMIN_PASSWORD)
                correct_otp = verify_totp(otp_candidate, secret) if otp_candidate else False

                if correct_user and correct_pass and correct_otp:
                    return u
        except Exception:
            pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized. Valid Admin credentials and 6-digit OTP required."
    )

class AdminLoginRequest(BaseModel):
    username: str = Field(..., max_length=100)
    password: str = Field(..., max_length=200)
    otp: str = Field(..., min_length=6, max_length=6)

@app.post("/api/manage/login")
async def admin_login(req: AdminLoginRequest, response: Response):
    correct_user = secrets.compare_digest(req.username, ADMIN_USERNAME)
    correct_pass = secrets.compare_digest(req.password, ADMIN_PASSWORD)
    secret = get_or_create_otp_secret()
    correct_otp = verify_totp(req.otp, secret)

    if not (correct_user and correct_pass and correct_otp):
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid username, password, or 6-digit OTP code."}
        )

    token = create_session_token(req.username)
    response.set_cookie(
        key="juris_admin_token",
        value=token,
        max_age=28800,
        httponly=True,
        samesite="lax"
    )
    return {"status": "authenticated", "token": token, "username": req.username}

@app.post("/api/manage/logout")
async def admin_logout(response: Response):
    response.delete_cookie("juris_admin_token")
    return {"status": "logged_out"}

@app.get("/api/manage/otp-setup")
async def get_otp_setup():
    secret = get_or_create_otp_secret()
    issuer = "Juris Philippine Legal AI"
    uri = f"otpauth://totp/{urllib.parse.quote(issuer)}:{ADMIN_USERNAME}?secret={secret}&issuer={urllib.parse.quote(issuer)}"
    return {
        "secret": secret,
        "username": ADMIN_USERNAME,
        "issuer": issuer,
        "otpauth_uri": uri,
        "format": "RFC 6238 TOTP (6-digit, 30s period)"
    }

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

# ==========================================
# LIVE DOCUMENT INGESTION & IMPORTER ENDPOINTS
# ==========================================

class IngestPreviewUrlRequest(BaseModel):
    url: str = Field(..., min_length=5, max_length=1000)

class IngestPreviewRawRequest(BaseModel):
    content: str = Field(..., min_length=20)
    is_html: Optional[bool] = False
    title: Optional[str] = None
    category: Optional[str] = None

class IngestCommitRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=400)
    category: str = Field(..., max_length=100)
    doc_type: Optional[str] = "Statute"
    doc_number: Optional[str] = ""
    year: Optional[int] = None
    date: Optional[str] = ""
    ponente: Optional[str] = ""
    source_url: Optional[str] = "manual_ingestion"
    full_text: str = Field(..., min_length=50)
    overwrite: Optional[bool] = False

class IngestCheckDuplicateRequest(BaseModel):
    doc_number: Optional[str] = ""
    title: Optional[str] = ""
    source_url: Optional[str] = ""

class IngestDeleteRequest(BaseModel):
    doc_id: Optional[str] = ""
    doc_number: Optional[str] = ""
    source_url: Optional[str] = ""

class IngestRecordRequest(BaseModel):
    doc_id: Optional[str] = ""
    doc_number: Optional[str] = ""
    source_url: Optional[str] = ""
    title: Optional[str] = ""

@app.post("/api/manage/ingest/preview-url")
async def preview_ingest_url(req: IngestPreviewUrlRequest, auth: str = Depends(verify_admin)):
    try:
        from legal_ingestion_service import LegalIngestionService
        service = LegalIngestionService()
        preview = service.preview_from_url(req.url)
        return preview
    except Exception as e:
        logger.error(f"Error in preview_ingest_url: {e}", exc_info=True)
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/manage/ingest/preview-raw")
async def preview_ingest_raw(req: IngestPreviewRawRequest, auth: str = Depends(verify_admin)):
    try:
        from legal_ingestion_service import LegalIngestionService
        service = LegalIngestionService()
        preview = service.preview_from_raw(
            content=req.content,
            is_html=req.is_html or False,
            title_hint=req.title or "",
            category_hint=req.category or ""
        )
        return preview
    except Exception as e:
        logger.error(f"Error in preview_ingest_raw: {e}", exc_info=True)
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/manage/ingest/check-duplicate")
async def check_document_duplicate(req: IngestCheckDuplicateRequest, auth: str = Depends(verify_admin)):
    try:
        from legal_ingestion_service import LegalIngestionService
        service = LegalIngestionService()
        res = service.check_existing_document(
            doc_number=req.doc_number or "",
            title=req.title or "",
            source_url=req.source_url or ""
        )
        return res
    except Exception as e:
        logger.error(f"Error in check_document_duplicate: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/manage/ingest/duplicates")
async def get_all_duplicates(auth: str = Depends(verify_admin)):
    try:
        from legal_ingestion_service import LegalIngestionService
        service = LegalIngestionService()
        clusters = service.scan_all_duplicates()
        return {"clusters": clusters, "total_clusters": len(clusters)}
    except Exception as e:
        logger.error(f"Error scanning duplicates: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/manage/ingest/record")
async def get_document_record(req: IngestRecordRequest, auth: str = Depends(verify_admin)):
    try:
        from legal_ingestion_service import LegalIngestionService
        service = LegalIngestionService()
        res = service.get_full_document_record(
            doc_id=req.doc_id or "",
            doc_number=req.doc_number or "",
            source_url=req.source_url or "",
            title=req.title or ""
        )
        return res
    except Exception as e:
        logger.error(f"Error in get_document_record: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/manage/ingest/delete")
async def delete_ingest_document(req: IngestDeleteRequest, auth: str = Depends(verify_admin)):
    try:
        from legal_ingestion_service import LegalIngestionService
        service = LegalIngestionService()
        res = service.delete_document_from_qdrant(
            doc_id=req.doc_id or "",
            doc_number=req.doc_number or "",
            source_url=req.source_url or ""
        )
        return res
    except Exception as e:
        logger.error(f"Error in delete_ingest_document: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/manage/ingest/commit")
async def commit_ingest_document(req: IngestCommitRequest, auth: str = Depends(verify_admin)):
    try:
        from legal_ingestion_service import LegalIngestionService
        service = LegalIngestionService()
        metadata = {
            "title": req.title.strip(),
            "category": req.category.strip(),
            "doc_type": req.doc_type or "Statute",
            "doc_number": req.doc_number or "",
            "year": req.year or time.gmtime().tm_year,
            "date": req.date or "",
            "ponente": req.ponente or "",
            "source_url": req.source_url or "manual_ingestion"
        }
        res = service.commit_document_to_qdrant(metadata, req.full_text, overwrite=req.overwrite or False)
        return res
    except Exception as e:
        logger.error(f"Error in commit_ingest_document: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/manage/ingest/history")
async def get_ingest_history(limit: int = 20, auth: str = Depends(verify_admin)):
    try:
        from legal_ingestion_service import LegalIngestionService
        service = LegalIngestionService()
        history = service.get_ingestion_history(limit=limit)
        return {"history": history}
    except Exception as e:
        logger.error(f"Error in get_ingest_history: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

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

# ==========================================
# GPU HYBRID LEGAL SEARCH ENDPOINT
# ==========================================

class LegalSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    category: Optional[str] = None
    top_k: int = Field(default=7, ge=1, le=20)
    score_threshold: float = Field(default=0.20, ge=0.0, le=1.0)

@app.post("/api/legal/search")
async def search_legal_authorities(req: LegalSearchRequest):
    """
    Asynchronous legal search endpoint utilizing GPU BGE-M3 + Qdrant RRF + Jina Cross-Encoder.
    """
    try:
        from legal_retrieval_engine import LegalRetrievalService
        service = LegalRetrievalService()
        chunks = await service.retrieve_and_rerank_legal_context(
            query=req.query,
            category_filter=req.category,
            top_k_candidates=30,
            final_top_k=req.top_k,
            score_threshold=req.score_threshold
        )
        return {
            "query": req.query,
            "total_retrieved": len(chunks),
            "results": [c.dict() for c in chunks]
        }
    except Exception as e:
        logger.error(f"Error in /api/legal/search: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ==========================================
# FEEDBACK & CAPTCHA ENDPOINTS
# ==========================================

CAPTCHA_SECRET = os.getenv("JURIS_CAPTCHA_SECRET", secrets.token_hex(16))

class FeedbackSubmissionRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=200)
    phone: Optional[str] = Field(default=None, max_length=50)
    category: Optional[str] = Field(default="General Feedback", max_length=100)
    subject: str = Field(..., min_length=3, max_length=200)
    message: str = Field(..., min_length=10, max_length=5000)
    captcha_token: str = Field(..., min_length=10, max_length=500)
    captcha_answer: str = Field(..., min_length=1, max_length=20)

@app.get("/api/feedback/captcha")
async def get_feedback_captcha():
    a = secrets.randbelow(18) + 3 # 3 to 20
    b = secrets.randbelow(15) + 1 # 1 to 15
    operator = secrets.choice(["+", "-"])
    
    if operator == "+":
        ans = a + b
        question = f"What is {a} + {b}?"
    else:
        if a < b:
            a, b = b, a
        ans = a - b
        question = f"What is {a} - {b}?"
        
    ts = str(int(time.time()))
    payload = f"{ans}:{ts}"
    sig = hmac.new(CAPTCHA_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    token = f"{ts}:{sig}"
    
    return {
        "question": question,
        "token": token
    }

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackSubmissionRequest, request: Request):
    email_clean = req.email.strip().lower()
    if "@" not in email_clean or "." not in email_clean or len(email_clean) < 5:
        return JSONResponse(status_code=400, content={"error": "A valid email address is mandatory."})

    try:
        ts_str, sig = req.captcha_token.split(":", 1)
        ts = int(ts_str)
        if time.time() - ts > 600:
            return JSONResponse(status_code=400, content={"error": "CAPTCHA challenge expired. Please refresh and try again."})
        
        ans_clean = str(req.captcha_answer).strip()
        expected_payload = f"{ans_clean}:{ts_str}"
        expected_sig = hmac.new(CAPTCHA_SECRET.encode(), expected_payload.encode(), hashlib.sha256).hexdigest()
        
        if not secrets.compare_digest(sig, expected_sig):
            return JSONResponse(status_code=400, content={"error": "Incorrect CAPTCHA answer. Please solve the verification challenge."})
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid CAPTCHA token format."})

    os.makedirs("logs", exist_ok=True)
    submission_entry = {
        "timestamp": time.time(),
        "date_str": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "client_ip": request.client.host if request.client else "unknown",
        "email": email_clean,
        "phone": req.phone.strip() if req.phone else None,
        "category": req.category,
        "subject": req.subject.strip(),
        "message": req.message.strip(),
        "user_agent": request.headers.get("user-agent", "unknown")
    }

    try:
        with open("logs/feedback_submissions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(submission_entry, ensure_ascii=False) + "\n")
        logger.info(f"Received feedback submission from {email_clean} (subject: {req.subject[:30]})")
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return JSONResponse(status_code=500, content={"error": "Could not record feedback submission."})

    return {
        "status": "success",
        "message": "Thank you! Your feedback has been securely submitted."
    }

@app.get("/feedback", response_class=HTMLResponse)
async def read_feedback():
    with open("static/feedback.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/manage-juris", response_class=HTMLResponse)
async def read_manage_juris():
    with open("static/admin.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=9010, reload=False)
