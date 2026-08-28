# legal_retrieval_engine.py
"""
Anchor Inc. - Sovereign Legal AI Systems
Module: legal_retrieval_engine.py
Purpose: GPU-Accelerated BGE-M3 Hybrid Embeddings + Qdrant RRF + Jina Cross-Encoder
Target Hardware: NVIDIA GeForce RTX 5070 Ti (16GB VRAM, CUDA 12.x)
"""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from qdrant_client import AsyncQdrantClient, models

logger = logging.getLogger("juris.retrieval")

# =====================================================================
# DATA CONTRACTS
# =====================================================================

class RetrievedChunk(BaseModel):
    id: str | int
    doc_id: Optional[str] = None
    title: str = "Philippine Legal Authority"
    category: str = "Philippine Law"
    gr_no: Optional[str] = None
    law_no: Optional[str] = None
    date: Optional[Any] = None
    ponente: Optional[str] = None
    doctrine_status: str = "good_law"
    text: str
    rrf_score: float = 0.0
    rerank_score: float = 0.0


# =====================================================================
# MODEL INITIALIZATION & INFERENCE (BGE-M3 + JINA / BGE RERANKER)
# =====================================================================

class LegalModelManager:
    """
    Singleton manager for GPU model memory allocation.
    Loads BGE-M3 and Cross-Encoder Reranker into dedicated CUDA VRAM partitions.
    Includes graceful fallbacks for local test/CPU environments.
    """
    _instance: Optional["LegalModelManager"] = None

    def __init__(
        self,
        bge_model_name: str = os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3"),
        reranker_model_name: str = os.getenv("RERANKER_MODEL_NAME", "jinaai/jina-reranker-v2-base-multilingual"),
        device: Optional[str] = None,
    ):
        if device is not None:
            self.device = device
        elif HAS_TORCH and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.bge_model_name = bge_model_name
        self.reranker_model_name = reranker_model_name
        
        self.bge_m3 = None
        self.reranker_model = None
        self.reranker_tokenizer = None
        self._fallback_ranker = None
        self._fallback_embedder = None

        logger.info(f"Initializing Neural Legal Models on device: {self.device}")
        self._init_models()

    def _init_models(self):
        # 1. Initialize BGE-M3
        try:
            from FlagEmbedding import BGEM3FlagModel
            logger.info(f"Loading BGE-M3 from {self.bge_model_name} on {self.device}...")
            self.bge_m3 = BGEM3FlagModel(
                self.bge_model_name,
                use_fp16=(self.device == "cuda"),
                device=self.device
            )
            logger.info("BGE-M3 hybrid encoder successfully initialized.")
        except Exception as e:
            logger.warning(f"FlagEmbedding BGE-M3 initialization bypassed ({e}). Utilizing fallback sparse/dense embedder.")
            self.bge_m3 = None
            try:
                from fastembed import SparseTextEmbedding
                self._fallback_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
            except Exception as fe_err:
                logger.debug(f"FastEmbed sparse embedder error: {fe_err}")

        # 2. Initialize Jina / BGE Cross-Encoder Reranker
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            logger.info(f"Loading Cross-Encoder Reranker: {self.reranker_model_name} on {self.device}...")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                self.reranker_model_name,
                trust_remote_code=True
            )
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(self.device).eval()
            logger.info(f"Cross-Encoder ({self.reranker_model_name}) successfully loaded into VRAM.")
        except Exception as e:
            logger.warning(f"Cross-Encoder initialization bypassed ({e}). Falling back to FlashRank.")
            self.reranker_model = None
            self.reranker_tokenizer = None
            try:
                from flashrank import Ranker
                self._fallback_ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
            except Exception as fr_err:
                logger.debug(f"FlashRank fallback error: {fr_err}")

    @classmethod
    def get_instance(cls) -> "LegalModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode_query_bge_m3(self, query: str) -> Tuple[List[float], List[int], List[float]]:
        """
        Executes single forward pass on BGE-M3.
        Returns:
            dense_vector: 1024-dimensional normalized float list
            sparse_indices: Token IDs
            sparse_values: Lexical weights
        """
        if self.bge_m3 is not None:
            output = self.bge_m3.encode(
                [query],
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False
            )
            dense_vec = output["dense_vecs"][0].tolist()
            raw_lexical_weights = output["lexical_weights"][0]
            sparse_indices: List[int] = []
            sparse_values: List[float] = []

            for token_id_str, weight in raw_lexical_weights.items():
                sparse_indices.append(int(token_id_str))
                sparse_values.append(float(weight))

            return dense_vec, sparse_indices, sparse_values
        else:
            # Fallback embedding generation using Ollama dense + FastEmbed BM25 sparse
            dense_vec = []
            sparse_indices = []
            sparse_values = []
            try:
                from langchain_ollama import OllamaEmbeddings
                dense_emb = OllamaEmbeddings(model=os.getenv("EMBED_MODEL", "nomic-embed-text:latest"), base_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"))
                dense_vec = dense_emb.embed_query(query)
            except Exception as d_err:
                logger.debug(f"Ollama fallback dense embedder error: {d_err}")
                dense_vec = [0.0] * 768

            try:
                from fastembed import SparseTextEmbedding
                if self._fallback_embedder is None:
                    self._fallback_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
                sparse_res = list(self._fallback_embedder.embed(query))
                if sparse_res:
                    sparse_indices = sparse_res[0].indices.tolist()
                    sparse_values = sparse_res[0].values.tolist()
            except Exception as s_err:
                logger.debug(f"FastEmbed fallback sparse embedder error: {s_err}")
                sparse_indices = [abs(hash(w)) % 100000 for w in query.lower().split()]
                sparse_values = [1.0] * len(sparse_indices)

            return dense_vec, sparse_indices, sparse_values

    def rerank_pairs(self, query: str, candidate_texts: List[str]) -> List[float]:
        """
        Computes deep cross-attention similarity scores for query-passage pairs on GPU.
        """
        if not candidate_texts:
            return []

        if self.reranker_model is not None and self.reranker_tokenizer is not None:
            pairs = [[query, text] for text in candidate_texts]
            with torch.inference_mode():
                if hasattr(self.reranker_model, "compute_score"):
                    scores = self.reranker_model.compute_score(pairs, max_length=1024)
                else:
                    inputs = self.reranker_tokenizer(
                        pairs,
                        padding=True,
                        truncation=True,
                        max_length=1024,
                        return_tensors="pt"
                    ).to(self.device)
                    logits = self.reranker_model(**inputs).logits
                    if logits.shape[-1] == 1:
                        scores = torch.sigmoid(logits.squeeze(-1)).cpu().tolist()
                    else:
                        scores = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()

            return scores if isinstance(scores, list) else scores.tolist()
        elif self._fallback_ranker is not None:
            from flashrank import RerankRequest
            passages = [{"id": i, "text": t} for i, t in enumerate(candidate_texts)]
            req = RerankRequest(query=query, passages=passages)
            results = self._fallback_ranker.rerank(req)
            score_map = {item["id"]: float(item["score"]) for item in results}
            return [score_map.get(i, 0.0) for i in range(len(candidate_texts))]
        else:
            # Heuristic similarity fallback
            q_words = set(query.lower().split())
            scores = []
            for t in candidate_texts:
                t_words = set(t.lower().split())
                overlap = len(q_words.intersection(t_words)) / max(len(q_words), 1)
                scores.append(round(overlap, 4))
            return scores


# =====================================================================
# ASYNC RETRIEVAL & RERANKING SERVICE
# =====================================================================

class LegalRetrievalService:
    def __init__(
        self,
        qdrant_url: str = "http://127.0.0.1:6333",
        collection_name: str = "philippine_law",
        model_manager: Optional[LegalModelManager] = None
    ):
        self.client = AsyncQdrantClient(url=qdrant_url, timeout=10.0)
        self.collection_name = collection_name
        self.models = model_manager or LegalModelManager.get_instance()

    async def retrieve_and_rerank_legal_context(
        self,
        query: str,
        category_filter: Optional[str] = None,
        top_k_candidates: int = 30,
        final_top_k: int = 4,
        score_threshold: float = 0.20
    ) -> List[RetrievedChunk]:
        """
        End-to-End Asynchronous Retrieval Pipeline:
        1. Encodes query into dense & sparse vectors via BGE-M3 (offloaded to threadpool).
        2. Executes native Qdrant Hybrid Search with Reciprocal Rank Fusion (RRF).
        3. Re-ranks top-30 candidates on GPU with Jina Reranker.
        4. Filters and returns verified top-4 legal chunks.
        """
        # -------------------------------------------------------------
        # STEP 1: Single-Pass BGE-M3 Hybrid Query Embedding
        # -------------------------------------------------------------
        dense_vector, sparse_indices, sparse_values = await asyncio.to_thread(
            self.models.encode_query_bge_m3,
            query
        )

        # -------------------------------------------------------------
        # STEP 2: Native Qdrant Hybrid Search (Dense + Sparse with RRF)
        # -------------------------------------------------------------
        qdrant_filter: Optional[models.Filter] = None
        if category_filter and category_filter.strip():
            cat_clean = category_filter.strip()
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=cat_clean)
                    )
                ]
            )

        try:
            # Execute hybrid search utilizing Qdrant v1.13+ Universal Query API
            prefetch_list = []
            if dense_vector and any(v != 0.0 for v in dense_vector):
                prefetch_list.append(
                    models.Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=top_k_candidates,
                        filter=qdrant_filter,
                    )
                )
            if sparse_indices and sparse_values:
                prefetch_list.append(
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                        ),
                        using="sparse",
                        limit=top_k_candidates,
                        filter=qdrant_filter,
                    )
                )

            if prefetch_list:
                hybrid_response = await self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=prefetch_list,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k_candidates,
                    with_payload=True,
                    with_vectors=False,
                )
                raw_points = hybrid_response.points
            else:
                # Fallback scroll if vector parameters are empty
                scroll_res, _ = await self.client.scroll(
                    collection_name=self.collection_name,
                    limit=top_k_candidates,
                    with_payload=True
                )
                raw_points = scroll_res

        except Exception as q_err:
            logger.warning(f"Qdrant hybrid query fallback triggered ({q_err})")
            scroll_res, _ = await self.client.scroll(
                collection_name=self.collection_name,
                limit=top_k_candidates,
                with_payload=True
            )
            raw_points = scroll_res

        if not raw_points:
            logger.warning(f"No candidate documents retrieved for query: '{query}'")
            return []

        # Parse retrieved points into working models
        candidates: List[RetrievedChunk] = []
        candidate_texts: List[str] = []

        for p in raw_points:
            payload = p.payload or {}
            chunk_text = payload.get("text", "").strip()
            if not chunk_text:
                continue

            chunk = RetrievedChunk(
                id=str(p.id),
                doc_id=payload.get("doc_id"),
                title=payload.get("title", "Philippine Legal Authority"),
                category=payload.get("category", "Philippine Law"),
                gr_no=payload.get("gr_no"),
                law_no=payload.get("law_no"),
                date=payload.get("date") or payload.get("year"),
                ponente=payload.get("ponente"),
                doctrine_status=payload.get("doctrine_status", "good_law"),
                text=chunk_text,
                rrf_score=float(getattr(p, "score", 0.0) or 0.0)
            )
            candidates.append(chunk)
            candidate_texts.append(chunk_text)

        # -------------------------------------------------------------
        # STEP 3: Jina GPU Cross-Encoder Deep Semantic Re-ranking
        # -------------------------------------------------------------
        rerank_scores = await asyncio.to_thread(
            self.models.rerank_pairs,
            query,
            candidate_texts
        )

        for chunk, score in zip(candidates, rerank_scores):
            chunk.rerank_score = float(score)

        # Sort by Cross-Encoder score descending
        candidates.sort(key=lambda x: x.rerank_score, reverse=True)

        # -------------------------------------------------------------
        # STEP 4: Pruning & Score Threshold Filtering
        # -------------------------------------------------------------
        verified_chunks = [
            c for c in candidates if c.rerank_score >= score_threshold
        ][:final_top_k]

        # Fallback safeguard: Return highest scoring candidate if all fall below threshold
        if not verified_chunks and candidates:
            verified_chunks = [candidates[0]]

        logger.info(
            f"Query: '{query[:40]}...' | Evaluated: {len(candidates)} candidates | "
            f"Returned Top {len(verified_chunks)} (Best Score: {verified_chunks[0].rerank_score:.4f})"
        )

        return verified_chunks
