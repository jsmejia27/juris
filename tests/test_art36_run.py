import requests, json

def run_live_test():
    query = 'Under Philippine law, is a clinical or psychiatric evaluation mandatory to prove psychological incapacity for the declaration of nullity of marriage under Article 36 of the Family Code? Cite the controlling jurisprudence.'

    url = 'http://127.0.0.1:9010/api/chat/stream'
    payload = {
        'message': query,
        'history': [],
        'model': 'qwen3.5:9b',
        'temperature': 0.0,
        'num_ctx': 16384,
        'top_k': 8
    }

    resp = requests.post(url, json=payload, stream=True)
    full_text = ''
    thought = ''
    sources = []
    verification = None

    for line in resp.iter_lines():
        if not line: continue
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data = json.loads(line_str[6:])
            t = data.get('type')
            if t == 'token':
                full_text += data.get('token', '')
            elif t == 'thought':
                thought += data.get('thought', '')
            elif t == 'sources':
                sources = data.get('sources', [])
            elif t == 'verification':
                verification = data.get('summary')

    output_data = {
        "query": query,
        "sources": sources,
        "response": full_text,
        "verification": verification
    }

    with open("eval/art36_output.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("Saved output to eval/art36_output.json")
    print("Verification summary:", json.dumps(verification, indent=2))

if __name__ == '__main__':
    run_live_test()

