"""DiffBot Analyze API — batch scraping URLi które trafilatura odrzuciła.

Tryb pracy:
  1. Wczytuje URLi z `errors.jsonl` z poprzedniej iteracji scrapingu.
  2. Dla każdego URL woła DiffBot Analyze API (https://api.diffbot.com/v3/analyze).
  3. Parsuje response do ujednoliconego formatu (lang, title, text, html, author, date).
  4. Zapisuje sukces do `scraped_diffbot.jsonl`, fail do `errors_diffbot.jsonl`.
  5. Liczy koszt (DiffBot bills per request: $0.005-0.02 dep. plan).

Limity bezpieczeństwa:
  - --max-calls: twardy limit liczby wywołań (default 6000, bufor poniżej 10k trial).
  - --rate-limit-per-min: throttling (default 60 req/min).
  - --resume: wznawia od ostatniego URL w output (idempotent).

Wymaga: DIFFBOT_TOKEN w .env lub envvar.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("diffbot_scrape")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class DiffBotResult:
    url: str
    success: bool
    title: str = ""
    text: str = ""
    lang: str = ""
    html: str = ""
    author: str = ""
    date: str = ""
    error: str = ""
    raw: dict[str, Any] | None = None


def _parse_authors(obj: dict[str, Any]) -> str:
    """Bezpieczne wyciągnięcie autorów — DiffBot zwraca listę stringów LUB dictów {name:...}."""
    authors = obj.get("authors")
    if isinstance(authors, list):
        names = []
        for a in authors:
            if isinstance(a, dict):
                names.append(str(a.get("name", "")))
            else:
                names.append(str(a))
        return ", ".join(n for n in names if n)
    if authors:
        return str(authors)
    return str(obj.get("author", "") or "")


class DiffBotClient:
    BASE_URL = "https://api.diffbot.com/v3/analyze"
    DEFAULT_TIMEOUT = 60

    def __init__(self, token: str, timeout: int = DEFAULT_TIMEOUT):
        if not token:
            raise ValueError("DIFFBOT_TOKEN missing — set in .env or env")
        self.token = token
        self.timeout = timeout
        self.session = requests.Session()

    def analyze(self, url: str, max_retries: int = 4) -> DiffBotResult:
        params = {"token": self.token, "url": url, "discussion": "false"}
        # Retry z exponential backoff na 429 (rate limit) i 503 (server busy)
        backoffs = [5, 15, 40, 90]  # sekundy
        for attempt in range(max_retries + 1):
            try:
                resp = self.session.get(f"{self.BASE_URL}?{urlencode(params)}", timeout=self.timeout)
                if resp.status_code in (429, 503):
                    if attempt < max_retries:
                        import time as _t
                        wait = backoffs[min(attempt, len(backoffs) - 1)]
                        _t.sleep(wait)
                        continue
                    return DiffBotResult(url=url, success=False,
                                          error=f"HTTP {resp.status_code} po {max_retries} retry: {resp.text[:120]}")
                if resp.status_code != 200:
                    return DiffBotResult(url=url, success=False,
                                          error=f"HTTP {resp.status_code}: {resp.text[:200]}")
                break
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    import time as _t
                    _t.sleep(backoffs[min(attempt, len(backoffs) - 1)])
                    continue
                return DiffBotResult(url=url, success=False, error="Timeout po retry")
            except Exception as e:
                return DiffBotResult(url=url, success=False, error=f"Exception: {e}")
        try:
            data = resp.json()
            if "error" in data:
                return DiffBotResult(url=url, success=False,
                                      error=f"DiffBot error: {data['error']}", raw=data)
            objs = data.get("objects", [])
            if not objs:
                return DiffBotResult(url=url, success=False, error="No objects in response", raw=data)
            obj = objs[0]
            text = obj.get("text", "")
            if not text or len(text) < 50:
                return DiffBotResult(url=url, success=False, error=f"Empty/too short text ({len(text)})", raw=data)
            return DiffBotResult(
                url=url,
                success=True,
                title=obj.get("title", ""),
                text=text,
                lang=obj.get("humanLanguage", "") or obj.get("naturalLanguage", [""])[0] if obj.get("naturalLanguage") else "",
                html=obj.get("html", ""),
                author=_parse_authors(obj),
                date=obj.get("date", "") or obj.get("estimatedDate", ""),
                raw=data,
            )
        except requests.exceptions.Timeout:
            return DiffBotResult(url=url, success=False, error="Timeout")
        except Exception as e:
            return DiffBotResult(url=url, success=False, error=f"Exception: {e}")


def _read_input_urls(path: Path) -> list[dict[str, Any]]:
    """Wczytaj URLi z errors.jsonl (lub innego JSONL)."""
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # Expect field 'article_url' lub 'url'
                url = item.get("article_url") or item.get("url")
                if url:
                    items.append({"url": url, "meta": item})
            except json.JSONDecodeError:
                continue
    return items


def _read_existing_output(path: Path) -> set[str]:
    """Zwraca URLi już obecne w output (idempotent resume)."""
    if not path.exists():
        return set()
    seen = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                seen.add(json.loads(line)["article_url"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL z URLi (errors.jsonl)")
    parser.add_argument("--output", required=True, help="JSONL output (scraped_diffbot.jsonl)")
    parser.add_argument("--errors-output", default=None,
                        help="JSONL z błędami (default: <output>.errors.jsonl)")
    parser.add_argument("--max-calls", type=int, default=6000,
                        help="Twardy limit liczby wywołań (default 6000)")
    parser.add_argument("--rate-limit-per-min", type=int, default=60,
                        help="Throttle (default 60 req/min)")
    parser.add_argument("--cost-per-call-usd", type=float, default=0.01,
                        help="Estimated cost per call (default $0.01)")
    parser.add_argument("--budget-usd", type=float, default=60.0,
                        help="Twardy limit kosztów USD (default $60)")
    parser.add_argument("--resume", action="store_true", help="Pomiń URLi już w output")
    parser.add_argument("--language-filter", default=None,
                        help="Comma-sep języki do scrapowania jeśli meta zawiera 'language' (np. 'pl,ru,uk')")
    args = parser.parse_args()

    token = os.getenv("DIFFBOT_TOKEN")
    if not token:
        logger.error("DIFFBOT_TOKEN missing in env")
        return 1

    client = DiffBotClient(token=token)
    inp = Path(args.input)
    out = Path(args.output)
    errors_out = Path(args.errors_output) if args.errors_output else out.with_suffix(".errors.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    urls = _read_input_urls(inp)
    logger.info("Loaded %d URLs from %s", len(urls), inp)

    if args.language_filter:
        langs = set(args.language_filter.split(","))
        before = len(urls)
        urls = [u for u in urls if u["meta"].get("language", "").lower() in langs]
        logger.info("Language filter %s: %d → %d", langs, before, len(urls))

    seen = _read_existing_output(out) if args.resume else set()
    logger.info("Already in output: %d (resume mode)", len(seen))

    todo = [u for u in urls if u["url"] not in seen][:args.max_calls]
    logger.info("To process: %d (cap %d)", len(todo), args.max_calls)

    if not todo:
        logger.info("Nothing to do.")
        return 0

    min_interval = 60.0 / args.rate_limit_per_min  # seconds between calls
    n_success = 0
    n_fail = 0
    cost_usd = 0.0

    with out.open("a", encoding="utf-8") as f_out, errors_out.open("a", encoding="utf-8") as f_err:
        for i, item in enumerate(todo):
            url = item["url"]
            meta = item["meta"]

            if cost_usd >= args.budget_usd:
                logger.warning("Budget reached ($%.2f >= $%.2f) — stopping", cost_usd, args.budget_usd)
                break

            t0 = time.time()
            result = client.analyze(url)
            elapsed = time.time() - t0

            cost_usd += args.cost_per_call_usd

            if result.success:
                row = {
                    "article_url": result.url,
                    "title": result.title,
                    "text": result.text,
                    "lang": result.lang,
                    "author": result.author,
                    "date": result.date,
                    "scraper": "diffbot",
                    # Zachowaj meta z input
                    "label": meta.get("label"),
                    "debunk_id": meta.get("debunk_id"),
                    "case_id": meta.get("case_id"),
                    "publication_date": meta.get("publication_date"),
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                f_out.flush()
                n_success += 1
            else:
                f_err.write(json.dumps({
                    "article_url": url, "error": result.error, "scraper": "diffbot",
                }, ensure_ascii=False) + "\n")
                f_err.flush()
                n_fail += 1

            if (i + 1) % 50 == 0:
                logger.info("Progress: %d/%d (✓ %d, ✗ %d) cost ~$%.2f",
                            i + 1, len(todo), n_success, n_fail, cost_usd)

            # Throttle
            sleep_for = max(0.0, min_interval - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

    logger.info("Done. Success: %d, Fail: %d, Cost: ~$%.2f", n_success, n_fail, cost_usd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
