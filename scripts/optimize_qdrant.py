import requests
import json
import time

url = 'http://127.0.0.1:6333/collections/philippine_law'

payload = {
    'optimizers_config': {
        'indexing_threshold': 1000
    }
}

print("Triggering Qdrant collection optimizer update...")
r = requests.patch(url, json=payload)
print("PATCH Status:", r.status_code, r.text)

print("Monitoring optimization progress...")
for i in range(30):
    time.sleep(2)
    resp = requests.get(url)
    if resp.ok:
        res = resp.json().get('result', {})
        status = res.get('status')
        opt_status = res.get('optimizer_status')
        indexed = res.get('indexed_vectors_count', 0)
        points = res.get('points_count', 0)
        segments = res.get('segments_count', 0)
        total_vectors = points * 2
        print(f"Step {i+1:02d}: Status={status.upper()} | Optimizer={opt_status} | Indexed={indexed:,} / {total_vectors:,} vectors | Segments={segments}")
        if status == 'green':
            print("\n SUCCESS: Collection is fully optimized and status is GREEN!")
            break
    else:
        print("Error fetching status:", resp.status_code)
