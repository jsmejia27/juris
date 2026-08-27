# open_congress.py
import logging
import requests
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

OPEN_CONGRESS_BASE_URL = "https://open-congress-api.bettergov.ph/api"

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by", "at",
    "from", "is", "are", "was", "were", "be", "been", "being", "what", "which", "who",
    "whom", "this", "that", "these", "those", "can", "could", "should", "would", "how",
    "why", "tell", "me", "about", "status", "legal", "law", "laws", "bill", "bills",
    "philippines", "philippine", "current", "pending", "regarding", "under", "according",
    "provisions", "rules", "republic", "act", "acts", "supreme", "court", "legislation",
    "legislative", "please", "discuss", "explain", "outline", "check", "find",
    # Tagalog/Filipino stopwords
    "ano", "mga", "ang", "ng", "sa", "nang", "mula", "para", "kay", "kina", "si", "sina",
    "ni", "nina", "tungkol", "ukol", "ipaliwanag", "tagalog", "filipino", "paano", "kailan",
    "sino", "bakit", "saan", "alin", "may", "meron", "wala", "ito", "iyon", "iyan", "lahat",
    "pwede", "maaari", "dapat", "sagutin", "sabihin", "alamin"
}

def extract_legal_keywords(query: str) -> List[str]:
    import re
    clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', query.lower())
    tokens = [w.strip() for w in clean.split() if len(w.strip()) > 2 and w.strip() not in STOPWORDS]
    candidates = []
    if len(tokens) >= 2:
        candidates.append(" ".join(tokens[-2:]))
        candidates.append(" ".join(tokens[:2]))
    if tokens:
        candidates.append(tokens[-1])
        for t in tokens[:-1]:
            candidates.append(t)
    return list(dict.fromkeys(candidates))

import concurrent.futures
import functools

class OpenCongressClient:
    def __init__(self, base_url: str = OPEN_CONGRESS_BASE_URL, default_timeout: float = 1.2):
        self.base_url = base_url.rstrip("/")
        self.default_timeout = default_timeout
        self._cache = {}

    def search_bills(
        self,
        query: str,
        limit_per_chamber: int = 2,
        congress: Optional[int] = 20
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Searches for both Senate Bills (SB) and House Bills (HB) matching the query concurrently.
        Returns a dictionary: {'senate_bills': [...], 'house_bills': [...]}
        """
        results = {
            "senate_bills": [],
            "house_bills": []
        }

        if not query or not query.strip():
            return results

        keywords = extract_legal_keywords(query)
        if not keywords:
            keywords = [query.strip()]

        kw = keywords[0]
        cache_key = f"{kw}_{congress}_{limit_per_chamber}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        def fetch_chamber(subtype: str, chamber_name: str, key_name: str):
            try:
                params = {
                    "q": kw,
                    "subtype": subtype,
                    "limit": limit_per_chamber
                }
                if congress:
                    params["congress"] = congress

                r = requests.get(
                    f"{self.base_url}/search/documents",
                    params=params,
                    timeout=self.default_timeout
                )
                if r.status_code == 200:
                    data = r.json().get("data", [])
                    parsed_list = []
                    for item in data:
                        parsed = self._parse_bill_record(item, chamber_name, subtype)
                        if parsed:
                            parsed_list.append(parsed)
                    return key_name, parsed_list
            except Exception as e:
                logger.debug(f"Open Congress API fast-timeout for '{kw}' {subtype}: {e}")
            return key_name, []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_sb = executor.submit(fetch_chamber, "SB", "Senate Bill", "senate_bills")
            future_hb = executor.submit(fetch_chamber, "HB", "House Bill", "house_bills")
            done, _ = concurrent.futures.wait([future_sb, future_hb], timeout=self.default_timeout + 0.3)
            for f in done:
                try:
                    k, bills = f.result()
                    results[k] = bills
                except Exception:
                    pass

        self._cache[cache_key] = results
        return results

    def _parse_bill_record(self, item: Dict[str, Any], chamber_label: str, subtype: str) -> Optional[Dict[str, Any]]:
        if not item or not isinstance(item, dict):
            return None

        name = item.get("name") or f"{subtype}-{item.get('bill_number', '')}"
        title = item.get("title") or item.get("congress_website_title") or item.get("long_title") or name
        long_title = item.get("long_title") or item.get("congress_website_title") or ""
        date_filed = item.get("date_filed") or ""
        congress_num = item.get("congress") or 20

        # Author extraction
        authors_list = []
        if item.get("authors_raw"):
            authors_list.append(str(item["authors_raw"]))
        elif item.get("authors") and isinstance(item["authors"], list):
            for auth in item["authors"]:
                if isinstance(auth, dict):
                    full_name = f"{auth.get('first_name', '')} {auth.get('last_name', '')}".strip()
                    if full_name:
                        authors_list.append(full_name)
        author_str = ", ".join(authors_list) if authors_list else "Not specified"

        # Download / Web link
        sources = item.get("download_url_sources") or []
        doc_url = sources[0] if sources else item.get("senate_website_permalink", "")

        return {
            "id": item.get("id", name),
            "bill_name": name,
            "bill_number": item.get("bill_number"),
            "chamber": chamber_label,
            "subtype": subtype,
            "category": "Senate Bill" if subtype == "SB" else "House Bill",
            "title": title.strip(),
            "long_title": long_title.strip(),
            "authors": author_str,
            "date_filed": date_filed,
            "subjects": item.get("subjects", []),
            "url": doc_url
        }

    def format_bills_context(self, bills_data: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Formats retrieved Senate & House bills into structured markdown text to inject into the LLM context.
        """
        senate_bills = bills_data.get("senate_bills", [])
        house_bills = bills_data.get("house_bills", [])

        if not senate_bills and not house_bills:
            return ""

        blocks = ["\n=== PENDING LEGISLATIVE INITIATIVES (PHILIPPINE CONGRESS & SENATE) ==="]
        blocks.append("Note: The following are active legislative bills and proposals filed in the Philippine Senate and House of Representatives:")

        counter = 1
        for sb in senate_bills:
            c_num = sb.get("congress", 20)
            blocks.append(
                f"\nPENDING BILL {counter}:\n"
                f"Chamber: Senate of the Philippines (Senate Bill)\n"
                f"Bill Identifier: {sb.get('bill_name', 'SB')} ({c_num}th Congress)\n"
                f"Short Title: {sb.get('title', '')}\n"
                f"Principal Authors/Sponsors: {sb.get('authors', 'Not specified')}\n"
                f"Date Filed: {sb.get('date_filed', 'N/A')}\n"
                f"Official Source URL: {sb.get('url') or 'https://web.senate.gov.ph'}\n"
                f"Full Statutory Summary: {sb.get('long_title') or sb.get('title', '')}"
            )
            counter += 1

        for hb in house_bills:
            c_num = hb.get("congress", 20)
            blocks.append(
                f"\nPENDING BILL {counter}:\n"
                f"Chamber: House of Representatives (House Bill)\n"
                f"Bill Identifier: {hb.get('bill_name', 'HB')} ({c_num}th Congress)\n"
                f"Short Title: {hb.get('title', '')}\n"
                f"Principal Authors/Sponsors: {hb.get('authors', 'Not specified')}\n"
                f"Date Filed: {hb.get('date_filed', 'N/A')}\n"
                f"Official Source URL: {hb.get('url') or 'https://www.congress.gov.ph'}\n"
                f"Full Statutory Summary: {hb.get('long_title') or hb.get('title', '')}"
            )
            counter += 1

        return "\n".join(blocks)
