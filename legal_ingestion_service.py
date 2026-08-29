# legal_ingestion_service.py
import os
import re
import json
import time
import uuid
import logging
import urllib.parse
from typing import Dict, Any, List, Optional
import requests
from bs4 import BeautifulSoup

from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import SparseTextEmbedding
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("legal_ingestion_service")

DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
COLLECTION_NAME = "philippine_law"
INGESTION_LOG_FILE = "logs/manual_ingested_documents.jsonl"
CHECKPOINT_FILE = "qdrant_server_data/ingested_docs.json"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 JurisLegalBot/1.0"
)

class LegalIngestionService:
    def __init__(self, qdrant_url: str = DEFAULT_QDRANT_URL, ollama_url: str = DEFAULT_OLLAMA_URL):
        self.qdrant_url = qdrant_url
        self.ollama_url = ollama_url
        self.client = QdrantClient(url=self.qdrant_url, timeout=30.0)
        
        # Dense Embedder
        self.embed_model = "nomic-embed-text"
        self.dense_embedder = OllamaEmbeddings(
            base_url=self.ollama_url,
            model=self.embed_model
        )
        
        # Sparse BM25 Embedder
        try:
            self.sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
        except Exception as e:
            logger.warning(f"Could not load FastEmbed BM25 sparse model: {e}")
            self.sparse_embedder = None

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=100,
            separators=["\n\nSection ", "\n\nARTICLE ", "\n\n", "\n", ". ", " "]
        )

    def fetch_url(self, url: str, timeout: int = 15) -> str:
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL format: '{url}'. Please provide a complete HTTP/HTTPS URL.")
        
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, verify=True)
        response.raise_for_status()
        
        # Fix character encoding issues common in LawPhil/Gov portals
        if response.encoding and response.encoding.lower() in ["iso-8859-1", "latin-1", "ascii"]:
            response.encoding = response.apparent_encoding or "utf-8"
            
        return response.text

    def clean_html_to_text(self, html_content: str, url: str = "") -> Dict[str, Any]:
        soup = BeautifulSoup(html_content, "html.parser")
        
        # 1. Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript", "form", "svg"]):
            element.decompose()
            
        # 2. Extract potential title from standard tags
        page_title = ""
        if soup.title and soup.title.string:
            page_title = soup.title.string.strip()

        # 3. Clean specific LawPhil / Official Gazette boilerplate
        # LawPhil specific cleaning:
        for font_tag in soup.find_all("font", {"size": ["1", "2"]}):
            text_val = font_tag.get_text().strip().lower()
            if "the lawphil project" in text_val or "arellano law foundation" in text_val:
                font_tag.decompose()
                
        for a_tag in soup.find_all("a"):
            if "lawphil.net" in (a_tag.get("href") or ""):
                # keep link text or decompose if it's navigation
                if a_tag.get_text().strip().lower() in ["home", "statutes", "jurisprudence", "executive issuances"]:
                    a_tag.decompose()

        # 4. Extract main readable text
        text_lines = []
        for p in soup.find_all(["p", "div", "blockquote", "h1", "h2", "h3", "h4", "h5", "li"]):
            txt = p.get_text().strip()
            if txt:
                text_lines.append(txt)
                
        full_text = "\n\n".join(text_lines)
        if not full_text.strip():
            # Fallback to direct soup text
            full_text = soup.get_text(separator="\n\n", strip=True)
            
        # Clean extra white spaces and duplicate newlines
        full_text = re.sub(r'\n{3,}', '\n\n', full_text).strip()
        
        return {
            "page_title": page_title,
            "cleaned_text": full_text,
            "char_count": len(full_text),
            "word_count": len(full_text.split())
        }

    def extract_legal_metadata(self, text: str, html_title: str = "", url: str = "") -> Dict[str, Any]:
        header_slice = text[:3500]
        url_lower = url.lower()
        
        # 1. Category Detection
        category = "Republic Act"
        doc_type = "Statute"
        
        if "/judjuris/" in url_lower or re.search(r'\b(G\.R\.\s*No\.|vs\.|P\s*O\s*N\s*E\s*N\s*T\s*E|EN\s+BANC)\b', header_slice, re.IGNORECASE):
            category = "Jurisprudence"
            doc_type = "Supreme Court Decision"
        elif "/execiss/" in url_lower or re.search(r'\b(EXECUTIVE\s+ORDER|PRESIDENTIAL\s+DECREE|ADMINISTRATIVE\s+ORDER|PROCLAMATION)\s+NO\.', header_slice, re.IGNORECASE):
            category = "Executive & Administrative Issuances"
            doc_type = "Executive Issuance"
        elif re.search(r'\b(CIRCULAR|MEMORANDUM\s+CIRCULAR|DEPARTMENT\s+ORDER)\s+NO\.', header_slice, re.IGNORECASE):
            category = "Executive & Administrative Issuances"
            doc_type = "Administrative Circular"
        elif re.search(r'\b(REPUBLIC\s+ACT|AN\s+ACT\s+(?:INSTITUTIONALIZING|CREATING|AMENDING|MANDATING|PROVIDING))\b', header_slice, re.IGNORECASE):
            category = "Republic Act"
            doc_type = "Statute"

        # 2. Extract Document Number
        doc_number = ""
        ra_match = re.search(r'\bREPUBLIC\s+ACT\s+(?:NO\.\s*)?([0-9]+)\b', header_slice, re.IGNORECASE)
        gr_match = re.search(r'\bG\.R\.\s*(?:No\.|Nos\.|L-)?\s*([A-Za-z0-9\-\,\s]+?)(?=\s+[A-Za-z]+|\n|\.|\,|$)', header_slice, re.IGNORECASE)
        eo_match = re.search(r'\b(?:EXECUTIVE\s+ORDER|PRESIDENTIAL\s+DECREE|ADMINISTRATIVE\s+ORDER|PROCLAMATION|CIRCULAR)\s+NO\.\s*([0-9\-\.A-Za-z]+)\b', header_slice, re.IGNORECASE)
        
        if ra_match:
            doc_number = f"RA {ra_match.group(1).strip()}"
        elif gr_match and category == "Jurisprudence":
            doc_number = gr_match.group(0).strip()
        elif eo_match:
            doc_number = eo_match.group(0).strip()

        # 3. Extract Promulgation / Enactment Date and Year
        date_match = re.search(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}', header_slice, re.IGNORECASE)
        date_str = date_match.group(0) if date_match else None
        
        year = None
        if date_str:
            y_match = re.search(r'\b(19\d\d|20\d\d)\b', date_str)
            if y_match:
                year = int(y_match.group(1))
                
        if not year:
            url_year = re.search(r'\b(19\d\d|20\d\d)\b', url)
            if url_year:
                year = int(url_year.group(1))
            else:
                y_text = re.search(r'\b(19\d\d|20\d\d)\b', header_slice)
                if y_text:
                    year = int(y_text.group(1))

        # 4. Extract Clean Title
        title = ""
        act_title_match = re.search(r'(AN\s+ACT\s+[A-Z0-9\s\,\-\;\:\(\)\–\.\/\'\"\n]+?)(?=\bBe\s+it\s+enacted\b|\bSection\s+1\b|\bApproved\s+(?:on)?\b|\n\n)', text, re.IGNORECASE)
        vs_match = re.search(r'([A-Z0-9\s\,\.\(\)\-\–]+?)\s*,\s*(?:petitioner|appellant|plaintiff|complainant|petitioner-appellant|accused-appellant)s?,\s*vs\.?\s*([A-Z0-9\s\,\.\(\)\-\–]+?)(?:,\s*(?:respondent|appellee|defendant|accused))', header_slice, re.IGNORECASE)
        
        if category == "Republic Act":
            if ra_match and act_title_match:
                clean_act = re.sub(r'\s+', ' ', act_title_match.group(1)).strip()
                title = f"Republic Act No. {ra_match.group(1)} – {clean_act}"[:250]
            elif act_title_match:
                title = re.sub(r'\s+', ' ', act_title_match.group(1)).strip()[:250]
            elif ra_match:
                title = f"Republic Act No. {ra_match.group(1)}"
            else:
                title = html_title or "Republic Act (Enacted Statute)"
        elif category == "Jurisprudence" and vs_match:
            p1 = re.sub(r'\s+', ' ', vs_match.group(1)).strip()
            p2 = re.sub(r'\s+', ' ', vs_match.group(2)).strip()
            title = f"{p1} vs. {p2}"[:220]
        else:
            title = html_title if html_title and len(html_title) > 5 else (text.split('\n')[0][:200] or "Philippine Legal Document")

        # 5. Extract Ponente (for Jurisprudence)
        ponente = None
        if category == "Jurisprudence":
            ponente_match = re.search(r'\*\*([A-Za-z\s\.\,\*\-]+?)(?:,\s*\*?(?:J\.|C\.J\.|JJ\.|Acting C\.J\.)\*?)\s*:\*\*', header_slice)
            if not ponente_match:
                ponente_match = re.search(r'(?:^|\n)\s*([A-Z][A-Z\s\.\-]{2,40})(?:,\s*(?:J\.|C\.J\.|JJ\.|Acting C\.J\.)(?:\s*:\s*)?)', header_slice)
            if ponente_match:
                ponente = ponente_match.group(1).replace('*', '').strip()

        return {
            "title": title,
            "category": category,
            "doc_type": doc_type,
            "doc_number": doc_number,
            "date": date_str,
            "year": year or time.gmtime().tm_year,
            "ponente": ponente,
            "source_url": url
        }

    def chunk_document(self, text: str) -> List[str]:
        raw_chunks = self.text_splitter.split_text(text)
        chunks = [c.strip() for c in raw_chunks if len(c.strip()) >= 50]
        return chunks

    def check_existing_document(self, doc_number: str = "", title: str = "", source_url: str = "") -> Dict[str, Any]:
        """
        Checks if a Republic Act, Jurisprudence ruling, or issuance is already indexed in Qdrant or logs.
        """
        existing_records = []
        matched_by = None
        
        # 1. Search in manual ingestion logs
        if os.path.exists(INGESTION_LOG_FILE):
            try:
                with open(INGESTION_LOG_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        e_doc_num = (entry.get("doc_number") or "").strip().lower()
                        e_url = (entry.get("source_url") or "").strip().lower()
                        e_title = (entry.get("title") or "").strip().lower()

                        q_doc_num = doc_number.strip().lower() if doc_number else ""
                        q_url = source_url.strip().lower() if source_url and source_url != "manual_ingestion" else ""
                        q_title = title.strip().lower() if title else ""

                        is_match = False
                        if q_doc_num and (q_doc_num == e_doc_num or (len(q_doc_num) >= 3 and q_doc_num in e_doc_num)):
                            is_match = True
                            matched_by = "doc_number"
                        elif q_url and q_url == e_url:
                            is_match = True
                            matched_by = "source_url"
                        elif q_title and len(q_title) > 10 and (q_title == e_title or q_title in e_title):
                            is_match = True
                            matched_by = "title"

                        if is_match:
                            existing_records.append({
                                "doc_id": entry.get("doc_id"),
                                "title": entry.get("title"),
                                "category": entry.get("category"),
                                "doc_number": entry.get("doc_number"),
                                "year": entry.get("year"),
                                "source_url": entry.get("source_url"),
                                "chunks_count": entry.get("chunks_count", 0),
                                "ingested_at": entry.get("ingested_at", "Previous Session"),
                                "source": "ingestion_log"
                            })
            except Exception as err:
                logger.warning(f"Error checking duplicate in ingestion log: {err}")

        # 2. Query Qdrant points directly if no log match or to get exact chunk count
        try:
            filter_conditions = []
            if source_url and source_url != "manual_ingestion":
                filter_conditions.append(models.FieldCondition(key="source_url", match=models.MatchValue(value=source_url)))
            if doc_number:
                filter_conditions.append(models.FieldCondition(key="doc_number", match=models.MatchValue(value=doc_number)))
                filter_conditions.append(models.FieldCondition(key="gr_no", match=models.MatchValue(value=doc_number)))
                # Extract digits if e.g. "RA 12254"
                ra_digits = re.search(r'\b(?:ra|republic\s+act(?:\s+no\.)?)\s*(\d+)\b', doc_number, re.IGNORECASE)
                if ra_digits:
                    d_num = ra_digits.group(1)
                    filter_conditions.append(models.FieldCondition(key="doc_number", match=models.MatchValue(value=f"RA {d_num}")))
                    filter_conditions.append(models.FieldCondition(key="gr_no", match=models.MatchValue(value=f"RA {d_num}")))
            if title and len(title) > 10:
                filter_conditions.append(models.FieldCondition(key="title", match=models.MatchValue(value=title)))

            if filter_conditions:
                scroll_filter = models.Filter(should=filter_conditions)
                res, _ = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=scroll_filter,
                    limit=30,
                    with_payload=True,
                    with_vectors=False
                )
                if res:
                    # Group by title or doc_id
                    found_doc_ids = set()
                    for pt in res:
                        p_load = pt.payload or {}
                        p_doc_id = p_load.get("doc_id") or p_load.get("title")
                        if p_doc_id and p_doc_id not in found_doc_ids:
                            found_doc_ids.add(p_doc_id)
                            # Check if already in existing_records
                            if not any(r.get("doc_id") == p_doc_id or r.get("title") == p_load.get("title") for r in existing_records):
                                existing_records.append({
                                    "doc_id": p_load.get("doc_id"),
                                    "title": p_load.get("title"),
                                    "category": p_load.get("category"),
                                    "doc_number": p_load.get("doc_number") or p_load.get("gr_no"),
                                    "year": p_load.get("year"),
                                    "source_url": p_load.get("source_url"),
                                    "chunks_count": p_load.get("total_chunks", len(res)),
                                    "ingested_at": p_load.get("ingested_at", "Indexed in Corpus"),
                                    "source": "qdrant_index"
                                })
                                if not matched_by:
                                    matched_by = "doc_number" if doc_number else ("source_url" if source_url else "title")
        except Exception as q_err:
            logger.debug(f"Direct Qdrant duplicate check notice: {q_err}")

        is_duplicate = len(existing_records) > 0
        return {
            "is_duplicate": is_duplicate,
            "exists": is_duplicate,
            "matched_by": matched_by,
            "match_count": len(existing_records),
            "existing_records": existing_records
        }

    def delete_document_from_qdrant(self, doc_id: str = "", doc_number: str = "", source_url: str = "") -> Dict[str, Any]:
        """
        Purges all vector chunks for a specific document from Qdrant and cleans up ingestion logs.
        """
        filter_conditions = []
        if doc_id:
            filter_conditions.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))
        if doc_number:
            filter_conditions.append(models.FieldCondition(key="doc_number", match=models.MatchValue(value=doc_number)))
            filter_conditions.append(models.FieldCondition(key="gr_no", match=models.MatchValue(value=doc_number)))
            ra_digits = re.search(r'\b(?:ra|republic\s+act(?:\s+no\.)?)\s*(\d+)\b', doc_number, re.IGNORECASE)
            if ra_digits:
                d_num = ra_digits.group(1)
                filter_conditions.append(models.FieldCondition(key="doc_number", match=models.MatchValue(value=f"RA {d_num}")))
                filter_conditions.append(models.FieldCondition(key="gr_no", match=models.MatchValue(value=f"RA {d_num}")))
        if source_url and source_url != "manual_ingestion":
            filter_conditions.append(models.FieldCondition(key="source_url", match=models.MatchValue(value=source_url)))

        if not filter_conditions:
            raise ValueError("At least one search parameter (doc_id, doc_number, or source_url) must be provided for deletion.")

        del_filter = models.Filter(should=filter_conditions)
        
        logger.info(f"Deleting matching vector points from Qdrant collection '{COLLECTION_NAME}'...")
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(filter=del_filter),
            wait=True
        )

        # Clean from INGESTION_LOG_FILE
        if os.path.exists(INGESTION_LOG_FILE):
            try:
                remaining_lines = []
                with open(INGESTION_LOG_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        e = json.loads(line)
                        match = False
                        if doc_id and e.get("doc_id") == doc_id:
                            match = True
                        if doc_number and e.get("doc_number") == doc_number:
                            match = True
                        if source_url and e.get("source_url") == source_url:
                            match = True
                        if not match:
                            remaining_lines.append(line)

                with open(INGESTION_LOG_FILE, "w", encoding="utf-8") as f:
                    f.writelines(remaining_lines)
            except Exception as log_err:
                logger.warning(f"Error updating ingestion log during deletion: {log_err}")

        return {
            "status": "success",
            "message": f"Successfully purged document from knowledge base (doc_number='{doc_number}', doc_id='{doc_id}')"
        }

    def scan_all_duplicates(self) -> List[Dict[str, Any]]:
        """
        Scans the manual ingestion logs and Qdrant points to identify duplicate document clusters.
        """
        duplicate_clusters = []
        if not os.path.exists(INGESTION_LOG_FILE):
            return duplicate_clusters

        records = []
        try:
            with open(INGESTION_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error scanning logs: {e}")
            return []

        # Group by doc_number
        by_doc_number = {}
        by_url = {}

        for r in records:
            d_num = (r.get("doc_number") or "").strip().lower()
            u = (r.get("source_url") or "").strip().lower()
            if d_num and d_num != "":
                by_doc_number.setdefault(d_num, []).append(r)
            if u and u not in ["", "manual_ingestion"]:
                by_url.setdefault(u, []).append(r)

        seen_keys = set()
        for k, group in by_doc_number.items():
            if len(group) > 1:
                seen_keys.add(k)
                duplicate_clusters.append({
                    "cluster_key": f"Document Number: {k.upper()}",
                    "cluster_type": "doc_number",
                    "duplicate_count": len(group),
                    "documents": group
                })

        for k, group in by_url.items():
            if len(group) > 1 and k not in seen_keys:
                duplicate_clusters.append({
                    "cluster_key": f"Source URL: {k}",
                    "cluster_type": "source_url",
                    "duplicate_count": len(group),
                    "documents": group
                })

        return duplicate_clusters

    def preview_from_url(self, url: str) -> Dict[str, Any]:
        html = self.fetch_url(url)
        cleaned = self.clean_html_to_text(html, url=url)
        metadata = self.extract_legal_metadata(cleaned["cleaned_text"], html_title=cleaned["page_title"], url=url)
        chunks = self.chunk_document(cleaned["cleaned_text"])
        
        # Check duplicate
        dup_check = self.check_existing_document(
            doc_number=metadata.get("doc_number") or "",
            title=metadata.get("title") or "",
            source_url=url
        )

        return {
            "metadata": metadata,
            "char_count": cleaned["char_count"],
            "word_count": cleaned["word_count"],
            "chunk_count": len(chunks),
            "sample_chunks": chunks[:3],
            "full_cleaned_text": cleaned["cleaned_text"],
            "duplicate_check": dup_check
        }

    def preview_from_raw(self, content: str, is_html: bool = False, title_hint: str = "", category_hint: str = "") -> Dict[str, Any]:
        if is_html:
            cleaned = self.clean_html_to_text(content)
            text = cleaned["cleaned_text"]
            html_title = cleaned["page_title"]
        else:
            text = content.strip()
            html_title = title_hint

        metadata = self.extract_legal_metadata(text, html_title=html_title)
        if category_hint:
            metadata["category"] = category_hint
        if title_hint:
            metadata["title"] = title_hint
            
        chunks = self.chunk_document(text)
        
        # Check duplicate
        dup_check = self.check_existing_document(
            doc_number=metadata.get("doc_number") or "",
            title=metadata.get("title") or "",
            source_url=metadata.get("source_url") or ""
        )

        return {
            "metadata": metadata,
            "char_count": len(text),
            "word_count": len(text.split()),
            "chunk_count": len(chunks),
            "sample_chunks": chunks[:3],
            "full_cleaned_text": text,
            "duplicate_check": dup_check
        }

    def commit_document_to_qdrant(self, metadata: Dict[str, Any], full_text: str, overwrite: bool = False) -> Dict[str, Any]:
        chunks = self.chunk_document(full_text)
        if not chunks:
            raise ValueError("No valid chunks could be created from the provided document text.")

        # If overwrite is requested, purge existing points for this document first
        if overwrite:
            try:
                self.delete_document_from_qdrant(
                    doc_number=metadata.get("doc_number") or "",
                    source_url=metadata.get("source_url") or ""
                )
            except Exception as del_err:
                logger.warning(f"Notice during overwrite purge: {del_err}")

        logger.info(f"Generating dense embeddings for {len(chunks)} chunks via {self.embed_model}...")
        dense_vectors = self.dense_embedder.embed_documents(chunks)
        
        sparse_vectors = None
        if self.sparse_embedder:
            try:
                sparse_vectors = list(self.sparse_embedder.embed(chunks))
            except Exception as e:
                logger.warning(f"FastEmbed BM25 sparse generation failed: {e}")

        doc_base_id = str(uuid.uuid4())
        points = []
        point_ids = []
        ingested_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        for idx, chunk_text in enumerate(chunks):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            vector_dict = {"dense": dense_vectors[idx]}
            if sparse_vectors and idx < len(sparse_vectors):
                sp = sparse_vectors[idx]
                vector_dict["sparse"] = models.SparseVector(
                    indices=sp.indices.tolist() if hasattr(sp.indices, 'tolist') else list(sp.indices),
                    values=sp.values.tolist() if hasattr(sp.values, 'tolist') else list(sp.values)
                )

            payload = {
                "doc_id": doc_base_id,
                "chunk_id": idx,
                "total_chunks": len(chunks),
                "title": metadata.get("title") or "Philippine Legal Document",
                "category": metadata.get("category") or "Republic Act",
                "doc_type": metadata.get("doc_type") or "Statute",
                "doc_number": metadata.get("doc_number") or "",
                "year": int(metadata.get("year") or time.gmtime().tm_year),
                "date": metadata.get("date") or "",
                "ponente": metadata.get("ponente") or "",
                "source_url": metadata.get("source_url") or "manual_ingestion",
                "ingested_at": ingested_at,
                "text": chunk_text
            }

            points.append(models.PointStruct(
                id=point_id,
                vector=vector_dict,
                payload=payload
            ))

        logger.info(f"Upserting {len(points)} vector points into Qdrant collection '{COLLECTION_NAME}'...")
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )

        os.makedirs(os.path.dirname(INGESTION_LOG_FILE) or ".", exist_ok=True)
        log_entry = {
            "doc_id": doc_base_id,
            "title": metadata.get("title"),
            "category": metadata.get("category"),
            "doc_number": metadata.get("doc_number"),
            "year": metadata.get("year"),
            "source_url": metadata.get("source_url"),
            "chunks_count": len(chunks),
            "point_ids": point_ids,
            "ingested_at": ingested_at
        }
        with open(INGESTION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        try:
            os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
            existing_ckpts = []
            if os.path.exists(CHECKPOINT_FILE):
                with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    existing_ckpts = json.load(f)
            existing_ckpts.append(doc_base_id)
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(existing_ckpts, f)
        except Exception as e:
            logger.warning(f"Could not update checkpoint file: {e}")

        return {
            "status": "success",
            "doc_id": doc_base_id,
            "title": metadata.get("title"),
            "category": metadata.get("category"),
            "chunks_indexed": len(chunks),
            "points_count": len(points),
            "ingested_at": ingested_at,
            "overwritten": overwrite
        }

    def get_ingestion_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not os.path.exists(INGESTION_LOG_FILE):
            return []
        
        entries = []
        try:
            with open(INGESTION_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error reading ingestion log file: {e}")
            return []
            
        return entries[::-1][:limit]
