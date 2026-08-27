# migrate_qdrant.py
import os
import time
import json
import shutil
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from ingest_data import init_qdrant_collection, DEFAULT_COLLECTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migration")

OLD_STORAGE_DIR = "./qdrant_storage"
NEW_STORAGE_DIR = "./qdrant_server_data"
NEW_QDRANT_URL = "http://127.0.0.1:6333"
BATCH_SIZE = 500

def migrate():
    logger.info("=== STARTING DIRECT POINT MIGRATION FROM EMBEDDED -> STANDALONE QDRANT ===")

    # 1. Copy Checkpoint File
    old_ckpt = os.path.join(OLD_STORAGE_DIR, "ingested_docs.json")
    new_ckpt = os.path.join(NEW_STORAGE_DIR, "ingested_docs.json")
    if os.path.exists(old_ckpt):
        os.makedirs(NEW_STORAGE_DIR, exist_ok=True)
        shutil.copy2(old_ckpt, new_ckpt)
        with open(new_ckpt, "r", encoding="utf-8") as f:
            docs = json.load(f)
        logger.info(f"✅ Migrated checkpoint: {len(docs)} parent documents tracked.")
    else:
        logger.warning(f"No checkpoint file found at {old_ckpt}")

    # 2. Connect to Old & New Clients
    logger.info(f"Connecting to Old Embedded Qdrant at {OLD_STORAGE_DIR}...")
    old_client = QdrantClient(path=OLD_STORAGE_DIR)

    logger.info(f"Connecting to New Standalone Qdrant Server at {NEW_QDRANT_URL}...")
    new_client = QdrantClient(url=NEW_QDRANT_URL, check_compatibility=False)

    # 3. Ensure collection exists on Standalone server
    init_qdrant_collection(new_client, DEFAULT_COLLECTION, vector_size=768)

    # 4. Check total points in old collection
    if not old_client.collection_exists(DEFAULT_COLLECTION):
        logger.error(f"Collection '{DEFAULT_COLLECTION}' not found in old storage.")
        return

    old_info = old_client.get_collection(DEFAULT_COLLECTION)
    total_old_points = old_info.points_count or 0
    logger.info(f"Old collection contains {total_old_points:,} vector points.")

    # 5. Scroll & Stream points in batches
    next_page_offset = None
    transferred_count = 0
    t0 = time.time()

    while True:
        records, next_page_offset = old_client.scroll(
            collection_name=DEFAULT_COLLECTION,
            limit=BATCH_SIZE,
            offset=next_page_offset,
            with_payload=True,
            with_vectors=True
        )

        if not records:
            break

        points_to_upsert = []
        for r in records:
            # Reconstruct PointStruct
            points_to_upsert.append(
                models.PointStruct(
                    id=r.id,
                    vector=r.vector,
                    payload=r.payload
                )
            )

        new_client.upsert(
            collection_name=DEFAULT_COLLECTION,
            points=points_to_upsert,
            wait=False
        )

        transferred_count += len(points_to_upsert)
        pct = (transferred_count / total_old_points * 100) if total_old_points > 0 else 100.0
        elapsed = time.time() - t0
        rate = transferred_count / elapsed if elapsed > 0 else 0
        logger.info(f"Transferred {transferred_count:,} / {total_old_points:,} points ({pct:.1f}%) [{rate:.0f} pts/sec]")

        if next_page_offset is None:
            break

    t_total = time.time() - t0
    new_info = new_client.get_collection(DEFAULT_COLLECTION)
    logger.info(f"🎉 MIGRATION COMPLETE in {t_total:.2f} seconds!")
    logger.info(f"New Standalone Qdrant Server points count: {new_info.points_count:,}")

if __name__ == "__main__":
    migrate()
