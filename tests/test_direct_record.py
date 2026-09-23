import sys
sys.path.insert(0, ".")
from legal_ingestion_service import LegalIngestionService

service = LegalIngestionService()
print("Checking duplicate for RA 386...")
dup = service.check_existing_document(doc_number="RA 386")
print("Duplicate result:", dup)

print("\nFetching full record for RA 386...")
rec = service.get_full_document_record(doc_number="RA 386")
print("Found:", rec.get("found"))
if rec.get("found"):
    print("Title:", rec["metadata"]["title"])
    print("Chunks count:", len(rec["chunks"]))
    print("Full text preview:", rec["full_text"][:200])
