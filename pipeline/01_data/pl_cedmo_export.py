"""CEDMO / Google Fact Check Tools API — pobranie polskich werdyktów.

CEDMO (Central European Digital Media Observatory) agreguje fact-checki polskich
organizacji (Demagog, AFP Sprawdzam, Stowarzyszenie...) i indeksuje je w
Google Fact Check Tools API:

  GET https://factchecktools.googleapis.com/v1alpha1/claims:search
      ?query=<keyword>&languageCode=pl&key=<API_KEY>

Wymagany API key: https://developers.google.com/fact-check/tools/api (free tier).
Zapisz jako `GOOGLE_FACTCHECK_API_KEY` w `.env`.

Response zawiera structured claims z polami:
  - text (claim verbatim)
  - claimReview[].title (artykuł fact-checkowy)
  - claimReview[].textualRating (werdykt, e.g. "Fałsz")
  - claimReview[].url
  - claimReview[].publisher.name

Strategia:
  1. Zapytanie dla każdego z N tematów ("Rosja", "Ukraina", "NATO", "wybory", "szczepionki", ...).
  2. Filter pl-language.
  3. Mapowanie textualRating → label binarna.
  4. Dedup po claim text + claimReview URL.

Brak rate limit Google (10k/dzień w trial).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("cedmo_export")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Polskie zapytania pokrywające szeroko polski landscape disinfo
QUERIES_PL = [
    "Rosja", "Ukraina", "Putin", "Zełeński", "NATO", "Unia Europejska", "wybory",
    "szczepionki", "COVID", "uchodźcy", "migracja", "Białoruś", "wojna",
    "Tusk", "Kaczyński", "Duda", "Komisja Europejska", "klimat", "gaz", "ropa",
    "dezinformacja", "fake news", "manipulacja", "propaganda", "kryzys",
]

# Mapping textualRating → binary label
TRUE_PATTERNS = ["prawda", "true", "correct", "potwierdzon", "prawdziwa"]
FALSE_PATTERNS = ["fałsz", "false", "fake", "manipulacja", "kłamst", "wprowadz", "nieprawda", "zmyślon"]


def _map_label(rating: str) -> int | None:
    r = (rating or "").lower()
    if any(p in r for p in FALSE_PATTERNS):
        return 1
    if any(p in r for p in TRUE_PATTERNS):
        return 0
    return None  # ambiguous (half-true, unverifiable) → drop


def _query_api(query: str, api_key: str, language: str = "pl", page_size: int = 50, max_pages: int = 5) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    page_token = ""
    for _ in range(max_pages):
        params = {
            "query": query,
            "languageCode": language,
            "pageSize": page_size,
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        url = f"{API_URL}?{urlencode(params)}"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                logger.warning("HTTP %d for query=%s: %s", resp.status_code, query, resp.text[:200])
                break
            data = resp.json()
            claims = data.get("claims", [])
            results.extend(claims)
            page_token = data.get("nextPageToken", "")
            if not page_token:
                break
        except Exception as e:
            logger.warning("Exception for query=%s: %s", query, e)
            break
    return results


def _claim_to_rows(claim: dict[str, Any]) -> list[dict[str, Any]]:
    """Z jednego claim potencjalnie wiele claimReview (różne organizacje weryfikowały)."""
    rows = []
    claim_text = claim.get("text", "")
    claimant = claim.get("claimant", "")
    claim_date = claim.get("claimDate", "")
    for cr in claim.get("claimReview", []):
        rating = cr.get("textualRating", "")
        label = _map_label(rating)
        if label is None:
            continue
        url = cr.get("url", "")
        if not url:
            continue
        publisher = cr.get("publisher", {}).get("name", "")
        # Filtruj tylko PL publisherów lub jawnie pl-language reviews
        if cr.get("languageCode", "pl") != "pl":
            continue
        # Tekst: claim + (jeśli short) tytuł review
        title = cr.get("title", "")
        text = claim_text
        if len(text) < 80 and title:
            text = f"{text} — {title}"
        if len(text) < 50:
            continue
        rows.append({
            "url": url,
            "title": title,
            "text": text,
            "label": label,
            "original_verdict": rating,
            "claimant": claimant,
            "publisher": publisher,
            "claim_date": claim_date,
            "review_date": cr.get("reviewDate", ""),
            "source": f"cedmo/{publisher.lower().replace(' ', '_') if publisher else 'unknown'}",
            "language": "pl",
            "fetched_at": datetime.utcnow().isoformat(),
            "license": "fair-use-google-fact-check-tools-api",
            "claim_id": f"cedmo_{hash(claim_text + url) & 0xFFFFFFFF:08x}",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="pl")
    parser.add_argument("--queries", default=None, help="Comma-sep, override default")
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=5)
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY")
    if not api_key:
        logger.error("GOOGLE_FACTCHECK_API_KEY missing in .env — pobierz na "
                     "https://developers.google.com/fact-check/tools/api")
        return 1

    queries = args.queries.split(",") if args.queries else QUERIES_PL
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Dedup
    seen_urls = set()
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    seen_urls.add(json.loads(line)["url"])
                except (json.JSONDecodeError, KeyError):
                    continue

    total_new = 0
    with out.open("a", encoding="utf-8") as f_out:
        for q in queries:
            logger.info("Query: %s", q)
            claims = _query_api(q, api_key, language=args.language,
                                page_size=args.page_size, max_pages=args.max_pages)
            logger.info("  → %d claims", len(claims))
            for claim in claims:
                for row in _claim_to_rows(claim):
                    if row["url"] in seen_urls:
                        continue
                    seen_urls.add(row["url"])
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_new += 1
            time.sleep(0.5)  # gentle

    logger.info("Total new rows: %d", total_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
