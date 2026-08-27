# ingest_data.py
import os
import re
import json
import uuid
import argparse
import logging
from typing import List, Dict, Any
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import SparseTextEmbedding
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_STORAGE_DIR = "./qdrant_server_data"
DEFAULT_COLLECTION = "philippine_law"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "qwen3-embedding:4b"
DEFAULT_EMBED_DIM = 2560
CHECKPOINT_FILE = "ingested_docs.json"

def init_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int = 2560):
    if not client.collection_exists(collection_name):
        logger.info(f"Creating Native Quantized Qdrant collection: {collection_name} (Dense dim: {vector_size}, INT8 Quantization, On-Disk mmap)")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                    on_disk=True
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True)
                )
            },
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=True
                )
            ),
            on_disk_payload=True
        )
    else:
        logger.info(f"Using existing Qdrant collection: {collection_name}")

def load_checkpoint(storage_dir: str) -> set:
    ckpt_path = os.path.join(storage_dir, CHECKPOINT_FILE)
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    return set()

def save_checkpoint(storage_dir: str, ingested_ids: set):
    os.makedirs(storage_dir, exist_ok=True)
    ckpt_path = os.path.join(storage_dir, CHECKPOINT_FILE)
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(list(ingested_ids), f)

def extract_case_metadata(text: str, title: str, basename: str, default_year: Any) -> Dict[str, Any]:
    lines = [l.strip() for l in text.split("\n") if l.strip()][:15]
    header = " \n ".join(lines)
    
    # 1. G.R. Number
    gr_match = re.search(r'G\.R\.\s*(?:No\.|Nos\.|L-)?\s*([A-Za-z0-9\-\,\s]+?)(?=\s+[A-Za-z]+|\n|\.|\,|$)', header, re.IGNORECASE)
    gr_no = gr_match.group(0).strip() if gr_match else basename
    
    # 2. Promulgation Date
    date_match = re.search(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}', header, re.IGNORECASE)
    date_str = date_match.group(0) if date_match else None
    
    # Year
    year = None
    if default_year and pd.notna(default_year):
        try:
            year = int(float(default_year))
        except Exception:
            pass
    if not year and date_str:
        y_match = re.search(r'\b(19\d\d|20\d\d)\b', date_str)
        if y_match:
            year = int(y_match.group(1))

    # 3. Ponente
    header_slice = text[:3000]
    ponente_match = re.search(r'\*\*([A-Z\s\.\,\*\-]+?)(?:,\s*\*?(?:J\.|C\.J\.|JJ\.|Acting C\.J\.)\*?)\s*:\*\*', header_slice)
    ponente = ponente_match.group(1).replace('*', '').strip() if ponente_match else None
    
    # 4. Title / Parties
    vs_match = re.search(r'([A-Z0-9\s\,\.\(\)\-\–]+?)\s*,\s*(?:petitioner|appellant|plaintiff|complainant|petitioner-appellant|accused-appellant)s?,\s*vs\.?\s*([A-Z0-9\s\,\.\(\)\-\–]+?)(?:,\s*(?:respondent|appellee|defendant|accused))', header_slice, re.IGNORECASE)
    if vs_match:
        case_title = f"{vs_match.group(1).strip()} vs. {vs_match.group(2).strip()}"[:200]
    else:
        case_title = title if (title and not title.lower().startswith('gr_')) else basename

    return {
        "gr_no": gr_no,
        "date": date_str,
        "year": year,
        "ponente": ponente,
        "title": case_title
    }

def is_valid_document(content: str, title: str, basename: str) -> bool:
    if not content or len(content.strip()) < 250:
        return False
    b_lower = str(basename).lower()
    t_lower = str(title).lower()
    if b_lower.endswith(('index', 'table', 'judjuris', 'repacts')) or t_lower.endswith(('index', 'table')):
        return False
    if re.match(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\d{4}$', b_lower):
        return False
    if content.count('|') > 30 and len(content.strip()) < 800:
        return False
    return True

def process_repacts_records(parquet_path: str, limit: int = None, exclude_ids: set = None) -> List[Dict[str, Any]]:
    logger.info(f"Loading Republic Acts from: {parquet_path}")
    pfile = pq.ParquetFile(parquet_path)
    records = []
    exclude_set = exclude_ids or set()
    
    for batch in pfile.iter_batches(batch_size=2000):
        for row in batch.to_pylist():
            if limit and len(records) >= limit:
                break
            content = str(row.get('content', '') or '')
            title = str(row.get('title', '') or '')
            basename = str(row.get('basename', '') or '')
            if not is_valid_document(content, title, basename):
                continue
            
            doc_id = str(row.get('id', '') or f'repacts_{basename}')
            if doc_id in exclude_set:
                continue

            ra_num = str(row.get('ra_bill_number', '') or '')
            summary = str(row.get('summary', '') or '')
            key_prov = str(row.get('key_provisions', '') or '')
            tags = str(row.get('tags', '') or '')
            keywords = str(row.get('keywords', '') or '')
            date_enacted = str(row.get('date_enacted', '') or '')
            year = None
            try:
                raw_y = row.get('year')
                year = int(float(raw_y)) if raw_y is not None else None
            except Exception:
                pass

            records.append({
                'doc_id': doc_id,
                'source': 'repacts',
                'category': 'Republic Act',
                'title': title or f'Republic Act No. {ra_num}',
                'gr_no': f'RA {ra_num}' if ra_num else basename,
                'date': date_enacted,
                'year': year,
                'ponente': str(row.get('principal_authors', '') or ''),
                'summary': summary,
                'key_provisions': key_prov,
                'tags': tags,
                'keywords': keywords,
                'content': content
            })
        if limit and len(records) >= limit:
            break
            
    logger.info(f"Loaded {len(records)} new/un-ingested Republic Acts records.")
    return records

def process_juris_records(parquet_path: str, limit: int = None, exclude_ids: set = None) -> List[Dict[str, Any]]:
    logger.info(f"Loading Jurisprudence cases from: {parquet_path}")
    pfile = pq.ParquetFile(parquet_path)
    records = []
    exclude_set = exclude_ids or set()
    cols = ['id', 'year', 'basename', 'title', 'content']
    
    for batch in pfile.iter_batches(batch_size=2000, columns=cols):
        for row in batch.to_pylist():
            if limit and len(records) >= limit:
                break
            content = str(row.get('content', '') or '')
            title = str(row.get('title', '') or '')
            basename = str(row.get('basename', '') or '')
            if not is_valid_document(content, title, basename):
                continue
            
            doc_id = str(row.get('id', '') or f'juris_{basename}')
            if doc_id in exclude_set:
                continue

            meta = extract_case_metadata(content, title, basename, row.get('year'))
            
            records.append({
                'doc_id': doc_id,
                'source': 'juris',
                'category': 'Jurisprudence',
                'title': meta['title'],
                'gr_no': meta['gr_no'],
                'date': meta['date'],
                'year': meta['year'],
                'ponente': meta['ponente'],
                'summary': '',
                'key_provisions': '',
                'tags': '',
                'keywords': '',
                'content': content
            })
        if limit and len(records) >= limit:
            break

    logger.info(f"Loaded {len(records)} new/un-ingested Jurisprudence records.")
    return records

def extract_canonical_basename(doc_id: str) -> str:
    if not doc_id:
        return ""
    b = os.path.basename(doc_id).replace('.md', '').replace('.txt', '')
    for prefix in ['repacts_', 'juris_', 'statutes_', 'executive_', 'consti_']:
        if b.lower().startswith(prefix):
            b = b[len(prefix):]
            break
    return b.lower().strip()

def process_consolidated_records(parquet_path: str, limit: int = None, exclude_ids: set = None, category_filter: str = None) -> List[Dict[str, Any]]:
    logger.info(f"Loading consolidated corpus records from: {parquet_path}")
    pfile = pq.ParquetFile(parquet_path)
    records = []
    exclude_set = exclude_ids or set()
    indexed_basenames = {extract_canonical_basename(i) for i in exclude_set}

    cols = ['id', 'source', 'category', 'year', 'month', 'path', 'basename', 'title', 'content']
    for batch in pfile.iter_batches(batch_size=2000, columns=cols):
        for row in batch.to_pylist():
            if limit and len(records) >= limit:
                break
                
            cat = str(row.get('category', '') or '').lower()
            if category_filter and category_filter != 'all' and cat != category_filter.lower():
                continue

            content = str(row.get('content', '') or '')
            title = str(row.get('title', '') or '')
            basename = str(row.get('basename', '') or '')
            if not is_valid_document(content, title, basename):
                continue
                
            raw_id = str(row.get('id', '') or f"{cat}_{basename}")
            can_base = extract_canonical_basename(basename or raw_id)

            if raw_id in exclude_set or can_base in indexed_basenames:
                continue

            if cat == 'juris':
                meta = extract_case_metadata(content, title, basename, row.get('year'))
                display_cat = "Jurisprudence"
                gr_no = meta['gr_no']
                doc_date = meta['date']
                year = meta['year']
                ponente = meta['ponente']
                clean_title = meta['title']
            elif cat == 'statutes':
                display_cat = "Statute / Republic Act"
                clean_title = title if title and not title.lower().startswith(('ra_', 'act_')) else f"Statute: {basename}"
                gr_no = basename.replace('_', ' ').upper()
                doc_date = ""
                year = row.get('year')
                ponente = ""
            elif cat == 'executive':
                display_cat = "Executive Issuance"
                clean_title = title if title and not title.lower().startswith(('eo_', 'pd_', 'ao_')) else f"Executive Issuance: {basename}"
                gr_no = basename.replace('_', ' ').upper()
                doc_date = ""
                year = row.get('year')
                ponente = ""
            elif cat == 'consti':
                display_cat = "Constitution"
                clean_title = title or "Constitution of the Republic of the Philippines"
                gr_no = "CONSTITUTION"
                doc_date = ""
                year = row.get('year')
                ponente = ""
            else:
                display_cat = "Philippine Law"
                clean_title = title or basename
                gr_no = ""
                doc_date = ""
                year = row.get('year')
                ponente = ""

            records.append({
                'doc_id': raw_id,
                'source': cat or 'consolidated',
                'category': display_cat,
                'title': clean_title,
                'gr_no': gr_no or '',
                'date': doc_date or '',
                'year': year,
                'ponente': ponente or '',
                'summary': '',
                'key_provisions': '',
                'tags': '',
                'keywords': '',
                'content': content
            })

        if limit and len(records) >= limit:
            break

    logger.info(f"Loaded {len(records)} new/un-ingested consolidated records.")
    return records

def split_legal_sections(text: str, category: str, max_chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Hierarchical legal chunker:
    1. Splits text along Section / Article boundaries or Case Digest markers.
    2. Sub-splits large sections while preserving Section context headers.
    """
    if "republic" in category.lower() or "statute" in category.lower() or "act" in category.lower():
        raw_sections = re.split(r'(?=\n(?:Section|Sec\.|Article|Art\.)\s+[0-9A-Za-z\-\(\)]+)', text, flags=re.IGNORECASE)
    elif "juris" in category.lower():
        raw_sections = re.split(r'(?=\n(?:FACTS|ISSUE|RULING|DOCTRINE|DECISION|FALLO|HELD)\b)', text, flags=re.IGNORECASE)
    else:
        raw_sections = text.split("\n\n")

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    final_chunks = []
    for sec in raw_sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= max_chunk_size:
            final_chunks.append(sec)
        else:
            sub_chunks = fallback_splitter.split_text(sec)
            final_chunks.extend(sub_chunks)

    return final_chunks if final_chunks else fallback_splitter.split_text(text)

def chunk_and_ingest(
    records: List[Dict[str, Any]],
    client: QdrantClient,
    collection_name: str,
    dense_embedder: OllamaEmbeddings,
    sparse_embedder: SparseTextEmbedding,
    storage_dir: str,
    ingested_ids: set,
    batch_size: int = 128
):
    all_chunks_payload = []
    all_chunks_text = []
    
    logger.info("Chunking documents with Section-Aware Hierarchical Chunker...")
    for rec in tqdm(records, desc="Chunking"):
        if rec['doc_id'] in ingested_ids:
            continue
        
        chunks = split_legal_sections(rec['content'], rec['category'])
        total_chunks = len(chunks)
        
        for idx, chunk in enumerate(chunks):
            contextualized_text = f"[{rec['category']}] {rec['title']} ({rec['gr_no']}): {chunk}"
            all_chunks_text.append(contextualized_text)
            all_chunks_payload.append({
                'doc_id': rec['doc_id'],
                'source': rec['source'],
                'category': rec['category'],
                'title': rec['title'],
                'gr_no': rec['gr_no'],
                'date': rec['date'],
                'year': rec['year'],
                'ponente': rec['ponente'],
                'summary': rec['summary'],
                'key_provisions': rec['key_provisions'],
                'tags': rec['tags'],
                'keywords': rec['keywords'],
                'chunk_index': idx + 1,
                'total_chunks': total_chunks,
                'text': chunk
            })
            
    total_chunks_to_ingest = len(all_chunks_text)
    logger.info(f"Total chunks to embed and index: {total_chunks_to_ingest}")
    if total_chunks_to_ingest == 0:
        logger.info("No new documents to ingest.")
        return

    points_buffer = []
    for i in tqdm(range(0, total_chunks_to_ingest, batch_size), desc="Embedding & Upserting"):
        batch_texts = all_chunks_text[i:i + batch_size]
        batch_payloads = all_chunks_payload[i:i + batch_size]
        
        dense_vectors = dense_embedder.embed_documents(batch_texts)
        sparse_vectors = list(sparse_embedder.embed(batch_texts))
        
        for text, payload, d_vec, s_vec in zip(batch_texts, batch_payloads, dense_vectors, sparse_vectors):
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    'dense': d_vec,
                    'sparse': models.SparseVector(
                        indices=s_vec.indices.tolist(),
                        values=s_vec.values.tolist()
                    )
                },
                payload=payload
            )
            points_buffer.append(point)
            
        client.upsert(collection_name=collection_name, points=points_buffer)
        points_buffer.clear()
        
        batch_doc_ids = {p['doc_id'] for p in batch_payloads}
        ingested_ids.update(batch_doc_ids)
        if (i // batch_size) % 10 == 0:
            save_checkpoint(storage_dir, ingested_ids)
            
    save_checkpoint(storage_dir, ingested_ids)
    logger.info("Ingestion complete! All chunks indexed in Qdrant.")

def main():
    parser = argparse.ArgumentParser(description="Philippine Legal Corpus Hybrid Ingestion into Qdrant")
    parser.add_argument("--source", type=str, choices=["repacts", "juris", "all"], default="repacts",
                        help="Data source to ingest (repacts, juris, or all)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of parent documents to process")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for vector generation and upsert (optimized for 16GB VRAM)")
    parser.add_argument("--storage-dir", type=str, default=DEFAULT_STORAGE_DIR, help="Directory for Qdrant storage")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="Qdrant collection name")
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL, help="Ollama base URL")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL, help="Ollama embedding model name")
    
    args = parser.parse_args()
    
    os.makedirs(args.storage_dir, exist_ok=True)
    ingested_ids = load_checkpoint(args.storage_dir)
    logger.info(f"Currently tracked ingested document count: {len(ingested_ids)}")
    
    client = QdrantClient(path=args.storage_dir)
    init_qdrant_collection(client, args.collection, vector_size=768)
    
    dense_embedder = OllamaEmbeddings(model=args.embed_model, base_url=args.ollama_url)
    sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    records = []
    if args.source in ["repacts", "all"]:
        repacts_path = "repacts_with_summary.parquet" if os.path.exists("repacts_with_summary.parquet") else "repacts.parquet"
        records.extend(process_repacts_records(repacts_path, limit=args.limit))
        
    if args.source in ["juris", "all"]:
        juris_path = "juris.parquet"
        records.extend(process_juris_records(juris_path, limit=args.limit))
        
    logger.info(f"Total records ready for indexing: {len(records)}")
    chunk_and_ingest(
        records=records,
        client=client,
        collection_name=args.collection,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        storage_dir=args.storage_dir,
        ingested_ids=ingested_ids,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
