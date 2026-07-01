"""OKO.press — best-effort scraping kategorii fact-check.

UWAGA: OKO.press to przede wszystkim dziennikarstwo śledcze, NIE klasyczny fact-check
z jednoznacznymi werdyktami. Skrypt zbiera artykuły z kategorii fact-check / sprawdzam
i próbuje wnioskować etykietę z tagów / tytułu. Wynik jest noisy i powinien być
**ostatnim wyborem** uzupełniającym po Demagog i CEDMO.

Strategia:
  1. RSS https://oko.press/category/fact-check/feed/ + ewentualnie inne taggowane sekcje.
  2. Parsowanie pojedynczego artykułu, wnioskowanie etykiety z heurystyk:
     - Tagi: 'fałsz', 'fakenews', 'manipulacja' → label=1
     - Tagi: 'prawda', 'potwierdzone' → label=0
     - Tytuł zawiera "FAŁSZ:" / "PRAWDA:" → odpowiednio
  3. Drop wszystko co nie da się jednoznacznie sklasyfikować.

Rate limit: 0.5 req/s.

Attribution: source="oko.press", oryginalny URL, license="fair-use-research".
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("oko_scrape")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


FEEDS = [
    "https://oko.press/category/fact-check/feed/",
    "https://oko.press/tag/fake-news/feed/",
    "https://oko.press/tag/dezinformacja/feed/",
]
USER_AGENT = "UMCS-DisinfoResearch/1.0 (academic; mailto:mbasarab@umcs.edu.pl)"


DISINFO_KEYWORDS = ["fałsz", "fake news", "fakenews", "manipulacja", "dezinformacj", "kłamst"]
TRUTH_KEYWORDS = ["prawda", "potwierdzon", "weryf"]


def _infer_label(title: str, tags: list[str], categories: list[str]) -> int | None:
    """Wnioskuje etykietę z tagów i tytułu. Zwraca None jeśli niejednoznaczne."""
    all_text = " ".join([title.lower(), *(t.lower() for t in tags), *(c.lower() for c in categories)])
    has_disinfo = any(k in all_text for k in DISINFO_KEYWORDS)
    has_truth = any(k in all_text for k in TRUTH_KEYWORDS)
    if has_disinfo and not has_truth:
        return 1
    if has_truth and not has_disinfo:
        return 0
    return None  # ambiguous → drop


def _fetch(url: str, timeout: int = 30) -> str | None:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


def _parse_article(html: str, url: str, feed_meta: dict) -> dict[str, Any] | None:
    soup = BeautifulSoup(html, "html.parser")

    title = soup.find("h1").get_text(strip=True) if soup.find("h1") else feed_meta.get("title", "")

    # Tags / categories — meta i breadcrumbs
    tags: list[str] = []
    for meta_tag in soup.find_all("meta", attrs={"property": "article:tag"}):
        if meta_tag.get("content"):
            tags.append(meta_tag["content"])
    categories: list[str] = []
    for cat in soup.select(".category, .breadcrumb a"):
        categories.append(cat.get_text(strip=True))

    label = _infer_label(title, tags, categories)
    if label is None:
        return None

    # Treść
    text_parts: list[str] = []
    for sel in [".entry-content", ".post-content", "article .content", "article"]:
        node = soup.select_one(sel)
        if node:
            for s in node(["script", "style", "aside", "nav", "footer", ".related", ".advertisement"]):
                s.decompose()
            text_parts.append(node.get_text(" ", strip=True))
            break

    text = " ".join(text_parts).strip()
    if len(text) < 100:
        return None

    return {
        "url": url,
        "title": title,
        "text": text,
        "label": label,
        "tags": tags,
        "categories": categories,
        "date": feed_meta.get("pub_date", ""),
        "source": "oko.press",
        "language": "pl",
        "fetched_at": datetime.utcnow().isoformat(),
        "license": "fair-use-research",
    }


def _entries_from_feeds(since: datetime | None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen = set()
    for feed_url in FEEDS:
        logger.info("Fetch feed: %s", feed_url)
        parsed = feedparser.parse(feed_url)
        for e in parsed.entries:
            url = e.get("link", "")
            if not url or url in seen:
                continue
            seen.add(url)
            try:
                pub_date = datetime(*e.published_parsed[:6]) if e.get("published_parsed") else None
            except Exception:
                pub_date = None
            if since and pub_date and pub_date < since:
                continue
            entries.append({
                "url": url,
                "title": e.get("title", ""),
                "pub_date": e.get("published", ""),
            })
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--since-date", default="2022-01-01")
    parser.add_argument("--max-entries", type=int, default=500)
    parser.add_argument("--rate-limit-per-sec", type=float, default=0.5)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    since = datetime.fromisoformat(args.since_date)

    entries = _entries_from_feeds(since=since)
    logger.info("Total RSS entries: %d", len(entries))

    seen = set()
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    seen.add(json.loads(line)["url"])
                except (json.JSONDecodeError, KeyError):
                    continue

    todo = [e for e in entries if e["url"] not in seen][:args.max_entries]
    n_ok = n_drop = n_fail = 0
    interval = 1.0 / args.rate_limit_per_sec

    with out.open("a", encoding="utf-8") as f_out:
        for i, e in enumerate(todo):
            t0 = time.time()
            html = _fetch(e["url"])
            if html is None:
                n_fail += 1
            else:
                parsed = _parse_article(html, e["url"], e)
                if parsed is None:
                    n_drop += 1
                else:
                    f_out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                    f_out.flush()
                    n_ok += 1
            time.sleep(max(0.0, interval - (time.time() - t0)))
            if (i + 1) % 25 == 0:
                logger.info("%d/%d ok=%d drop=%d fail=%d", i + 1, len(todo), n_ok, n_drop, n_fail)

    logger.info("Done. OK=%d drop=%d fail=%d", n_ok, n_drop, n_fail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
