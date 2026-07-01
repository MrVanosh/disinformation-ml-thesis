"""Demagog.org.pl — scraping werdyktów fact-check.

Strategia (zgodna z ToS, używamy publicznego RSS + parsowania HTML pojedynczych stron):
  1. Pobierz feedy RSS:
     - https://demagog.org.pl/feed/ (główny)
     - https://demagog.org.pl/wypowiedzi/feed/ (wypowiedzi polityków)
     - https://demagog.org.pl/fake_news/feed/ (fake newsy)
  2. Dla każdego entry idziemy na artykuł, parsujemy werdykt z HTML.
  3. Zapisujemy do JSONL z polami: url, title, text, label (binarna), original_verdict, date, source.

Werdykty Demagog mapowane do binarnej:
  - "Prawda"               → label=0 (not_disinfo)
  - "Częściowa prawda"     → label=0  (kontrowersyjne; może być excluded via flag)
  - "Manipulacja"          → label=1 (disinfo)
  - "Fałsz"                → label=1
  - "Nieweryfikowalne"     → dropped

Attribution: każdy wpis ma source="demagog.org.pl" oraz oryginalny URL — zgodnie z licencją
CC-BY (Demagog deklaruje fair use w celach badawczych).

Rate limit: 1 request/sek (uprzejmość dla serwera).
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

logger = logging.getLogger("demagog_scrape")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Tylko feedy z fact-checkami pojedynczych tez (ClaimReview JSON-LD).
# Pomijamy główny feed/ i analizy_i_raporty/ (długie analizy bez werdyktu pojedynczej tezy).
FEEDS = [
    "https://demagog.org.pl/wypowiedzi/feed/",
    "https://demagog.org.pl/fake_news/feed/",
]

USER_AGENT = "UMCS-DisinfoResearch/1.0 (academic; mailto:mbasarab@umcs.edu.pl)"

VERDICT_MAP = {
    # frazy dłuższe sprawdzane najpierw (sortowanie w _parse_article)
    "częściowy fałsz": ("disinfo", 1),
    "częściowa prawda": ("partial_truth", 0),  # neutralna, oznaczone osobno
    "wprowadza w błąd": ("disinfo", 1),
    "nieweryfikowalne": ("unverifiable", None),
    "zmanipulowane": ("disinfo", 1),
    "manipulacja": ("disinfo", 1),
    "prawda": ("not_disinfo", 0),
    "fałsz": ("disinfo", 1),
}


def _fetch(url: str, timeout: int = 30) -> str | None:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
        logger.debug("HTTP %d for %s", resp.status_code, url)
    except Exception as e:
        logger.debug("Error fetching %s: %s", url, e)
    return None


def _extract_claimreview(soup: BeautifulSoup) -> tuple[str, str]:
    """Wyciąga werdykt ze schema.org ClaimReview JSON-LD (najpewniejsze źródło).

    Zwraca (verdict_text_lower, claim_reviewed). Pusty string jeśli brak.
    Demagog/OKO i większość fact-checkerów osadza ClaimReview zgodnie ze standardem
    Google Fact Check (reviewRating.alternateName = "Fałsz"/"Prawda"/"Manipulacja"...).
    """
    import json as _json
    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            data = _json.loads(script.string)
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        # Czasem @graph zawiera listę
        for it in list(items):
            if isinstance(it, dict) and "@graph" in it:
                items.extend(it["@graph"])
        for it in items:
            if not isinstance(it, dict):
                continue
            if "ClaimReview" in str(it.get("@type", "")):
                rating = it.get("reviewRating", {}) or {}
                verdict = str(rating.get("alternateName", "")).strip().lower()
                claim = str(it.get("claimReviewed", "")).strip()
                if verdict:
                    return verdict, claim
    return "", ""


def _parse_article(html: str, url: str) -> dict[str, Any] | None:
    soup = BeautifulSoup(html, "html.parser")

    # Tytuł
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Werdykt: najpierw ClaimReview JSON-LD (pewne), potem fallback na CSS/regex
    verdict_text, claim_reviewed = _extract_claimreview(soup)
    if not verdict_text:
        for sel in [".verdict", ".rating", ".article-verdict", "[class*='verdict']", "[class*='rating']"]:
            node = soup.select_one(sel)
            if node:
                verdict_text = node.get_text(" ", strip=True).lower()
                break
    if not verdict_text:
        m = re.search(r"werdykt\s*:?\s*([^\n.]+)", soup.get_text(" ", strip=True), re.IGNORECASE)
        if m:
            verdict_text = m.group(1).lower()

    # Mapowanie werdyktu — sprawdzaj dłuższe frazy najpierw (np. "częściowa prawda" przed "prawda")
    label = None
    original_verdict = ""
    for needle, (verdict_str, lab) in sorted(VERDICT_MAP.items(), key=lambda kv: -len(kv[0])):
        if needle in verdict_text:
            label = lab
            original_verdict = verdict_str
            break

    if label is None:
        # Drop unverifiable / nieparsowalne (np. "częściowa prawda" jeśli odrzucamy neutralne)
        return None

    # ── KLUCZOWE: tekstem do klasyfikacji jest WERYFIKOWANA TEZA (claimReviewed),
    #    NIE treść artykułu Demagog (która jest rzetelnym debunkiem, nie dezinformacją).
    #    To czyni zbiór analogicznym do LIAR (krótkie stwierdzenie + werdykt).
    claim_text = (claim_reviewed or "").strip()
    if not claim_text:
        # Fallback: tytuł fake_news często JEST tezą; ale dla wypowiedzi tytuł to nie cytat
        claim_text = title.strip()
    if len(claim_text) < 20:
        # Za krótkie/puste — bez sensownej tezy do klasyfikacji
        return None

    # Treść artykułu (uzasadnienie Demagog) — zachowujemy OSOBNO jako kontekst/debunk,
    # NIGDY jako tekst do klasyfikacji.
    debunk_parts: list[str] = []
    for sel in [".entry-content", ".article-content", ".post-content", "article"]:
        node = soup.select_one(sel)
        if node:
            for s in node(["script", "style", "aside", "nav", "footer"]):
                s.decompose()
            debunk_parts.append(node.get_text(" ", strip=True))
            break
    debunk_text = " ".join(debunk_parts).strip()

    # Data
    date_str = ""
    date_tag = soup.find("time")
    if date_tag:
        date_str = date_tag.get("datetime", "") or date_tag.get_text(strip=True)

    # Autor / "Wypowiedź" - kogo dotyczy
    speaker = ""
    speaker_tag = soup.select_one(".speaker, .author-speaker, .article-speaker")
    if speaker_tag:
        speaker = speaker_tag.get_text(strip=True)

    return {
        "url": url,
        "title": title,
        "text": claim_text,            # weryfikowana TEZA (claimReviewed) — do klasyfikacji
        "debunk_text": debunk_text,    # uzasadnienie Demagog — kontekst, NIE do klasyfikacji
        "label": label,
        "original_verdict": original_verdict,
        "verdict_raw": verdict_text[:100],
        "date": date_str,
        "speaker": speaker,
        "source": "demagog.org.pl",
        "task_type": "claim",          # polski LIAR-like (claim-level), nie document-level
        "claim_id": url,               # group key dla split (jedna teza = jedna grupa)
        "language": "pl",
        "fetched_at": datetime.utcnow().isoformat(),
        "license": "CC-BY-research-use",
    }


def _entries_from_feeds(since: datetime | None) -> list[dict[str, Any]]:
    entries = []
    seen_urls = set()
    for feed_url in FEEDS:
        logger.info("Fetching feed: %s", feed_url)
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries:
            url = entry.get("link", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            pub_str = entry.get("published", entry.get("updated", ""))
            try:
                pub_date = datetime(*entry.published_parsed[:6]) if entry.get("published_parsed") else None
            except Exception:
                pub_date = None
            if since and pub_date and pub_date < since:
                continue
            entries.append({"url": url, "feed_title": entry.get("title", ""), "pub_date": pub_str})
    return entries


SITEMAP_INDEX = "https://demagog.org.pl/sitemap.xml"
# Tylko URLe fact-checków pojedynczych tez (ClaimReview). Demagog WordPress sitemap
# rozdziela typy postów na osobne sub-sitemapy.
SITEMAP_KEEP = ("fake_news", "wypowiedzi")


def _http_get(url: str, timeout: int = 30, retries: int = 3) -> str | None:
    """GET z retry/backoff — Demagog server bywa wolny."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            if r.status_code == 200:
                return r.text
        except Exception:
            pass
        time.sleep(3 * (attempt + 1))
    return None


def _entries_from_sitemap(since: datetime | None) -> list[dict[str, Any]]:
    """Pełne archiwum Demagog z sitemap (fake_news + wypowiedzi sub-sitemaps)."""
    idx_xml = _http_get(SITEMAP_INDEX)
    if not idx_xml:
        logger.warning("Sitemap index niedostępny — fallback na RSS")
        return _entries_from_feeds(since)
    idx = BeautifulSoup(idx_xml, "xml")
    sub_sitemaps = [loc.get_text() for loc in idx.find_all("loc")
                    if any(k in loc.get_text() for k in SITEMAP_KEEP)]
    logger.info("Sub-sitemaps do pobrania: %d", len(sub_sitemaps))

    entries = []
    seen = set()
    for sm in sub_sitemaps:
        sm_xml = _http_get(sm)
        if not sm_xml:
            logger.warning("Pominięto sub-sitemap (timeout): %s", sm)
            continue
        soup = BeautifulSoup(sm_xml, "xml")
        for url_tag in soup.find_all("url"):
            loc = url_tag.find("loc")
            if not loc:
                continue
            url = loc.get_text().strip()
            # tylko artykuły fact-check (nie strony kategorii/paginacji)
            if not any(f"/{k}/" in url for k in SITEMAP_KEEP):
                continue
            if url in seen:
                continue
            seen.add(url)
            lastmod_tag = url_tag.find("lastmod")
            pub_date = None
            pub_str = ""
            if lastmod_tag:
                pub_str = lastmod_tag.get_text().strip()
                try:
                    pub_date = datetime.fromisoformat(pub_str.replace("Z", "+00:00")).replace(tzinfo=None)
                except Exception:
                    pub_date = None
            if since and pub_date and pub_date < since:
                continue
            entries.append({"url": url, "feed_title": "", "pub_date": pub_str})
        logger.info("  %s → %d URLi (łącznie %d)", sm.split("/")[-1], len(entries), len(entries))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--since-date", default="2022-01-01", help="ISO date, skip entries before")
    parser.add_argument("--max-entries", type=int, default=2000)
    parser.add_argument("--rate-limit-per-sec", type=float, default=1.0)
    parser.add_argument("--source", choices=["sitemap", "rss"], default="sitemap",
                        help="sitemap = pełne archiwum (default); rss = tylko najnowsze")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    since = datetime.fromisoformat(args.since_date)
    if args.source == "sitemap":
        entries = _entries_from_sitemap(since=since)
    else:
        entries = _entries_from_feeds(since=since)
    logger.info("Total entries (%s): %d", args.source, len(entries))

    # Idempotent: wczytaj istniejące URLi
    seen = set()
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    seen.add(json.loads(line)["url"])
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info("Already in output: %d", len(seen))

    todo = [e for e in entries if e["url"] not in seen][:args.max_entries]
    logger.info("To process: %d", len(todo))

    n_ok = 0
    n_drop = 0
    n_fail = 0
    interval = 1.0 / args.rate_limit_per_sec

    with out.open("a", encoding="utf-8") as f_out:
        for i, e in enumerate(todo):
            url = e["url"]
            t0 = time.time()
            html = _fetch(url)
            if html is None:
                n_fail += 1
            else:
                parsed = _parse_article(html, url)
                if parsed is None:
                    n_drop += 1
                else:
                    f_out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                    f_out.flush()
                    n_ok += 1
            elapsed = time.time() - t0
            sleep_for = max(0.0, interval - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)
            if (i + 1) % 25 == 0:
                logger.info("Progress %d/%d ok=%d drop=%d fail=%d", i + 1, len(todo), n_ok, n_drop, n_fail)

    logger.info("Done. OK: %d, dropped: %d, failed: %d", n_ok, n_drop, n_fail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
