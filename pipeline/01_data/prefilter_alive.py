"""Pre-filtr żywych URLi przed DiffBot — oszczędza API calls na martwych linkach.

Lekki HTTP check (HEAD, fallback GET) każdego URLa z errors.jsonl. URLe zwracające
404/410/dead odrzucane PRZED kosztownym DiffBot. Opcjonalny filtr językowy.

Użycie:
    python prefilter_alive.py \\
        --input datasets/euvsdisinfo/errors.jsonl \\
        --output datasets/euvsdisinfo/errors_alive.jsonl \\
        --exclude-lang Arabic \\
        --workers 20
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

UA = "Mozilla/5.0 (Macintosh; Apple Silicon Mac OS X 14_0) AppleWebKit/605 UMCS-research"
# Kody traktowane jako "martwy link" — nie ma sensu próbować DiffBot
DEAD_CODES = {404, 410}


def check_url(item: dict, timeout: int = 8) -> tuple[dict, str]:
    """Zwraca (item, status) gdzie status ∈ {alive, dead, maybe}."""
    url = item.get("article_url") or item.get("url")
    headers = {"User-Agent": UA}
    try:
        # HEAD najpierw (lekki); fallback GET jeśli serwer nie wspiera HEAD
        r = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code == 405 or r.status_code >= 500:
            # HEAD niewspierany lub błąd serwera → spróbuj GET (stream, nie pobieraj body)
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
            r.close()
        if r.status_code in DEAD_CODES:
            return item, "dead"
        # 2xx/3xx → żywy; 401/403/paywall → "maybe" (DiffBot czasem przebije)
        if 200 <= r.status_code < 400:
            return item, "alive"
        return item, "maybe"
    except requests.exceptions.Timeout:
        return item, "maybe"  # timeout HEAD ≠ martwy; DiffBot ma własny retry
    except requests.exceptions.ConnectionError:
        return item, "dead"  # DNS fail / connection refused → raczej martwy
    except Exception:
        return item, "maybe"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--exclude-lang", action="append", default=[],
                   help="Język do pominięcia (można wielokrotnie). Np. --exclude-lang Arabic")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--timeout", type=int, default=8)
    args = p.parse_args()

    items = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            lang = str(d.get("language", ""))
            if lang in args.exclude_lang:
                continue
            items.append(d)

    print(f"Do sprawdzenia: {len(items)} URLi (po wykluczeniu {args.exclude_lang})", file=sys.stderr)

    alive, dead, maybe = [], 0, []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(check_url, it, args.timeout): it for it in items}
        for fut in as_completed(futures):
            item, status = fut.result()
            done += 1
            if status == "alive":
                alive.append(item)
            elif status == "maybe":
                maybe.append(item)
            else:
                dead += 1
            if done % 250 == 0:
                print(f"  [{done}/{len(items)}] alive={len(alive)} maybe={len(maybe)} dead={dead}",
                      file=sys.stderr)

    # alive + maybe idą do DiffBot (maybe bo paywall/timeout może się udać)
    keep = alive + maybe
    with open(args.output, "w", encoding="utf-8") as f:
        for item in keep:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Wynik pre-filtra:", file=sys.stderr)
    print(f"   alive: {len(alive)}", file=sys.stderr)
    print(f"   maybe (paywall/timeout — DiffBot spróbuje): {len(maybe)}", file=sys.stderr)
    print(f"   dead (404/410/DNS — pominięte): {dead}", file=sys.stderr)
    print(f"   → DiffBot dostanie: {len(keep)} URLi (zamiast {len(items)})", file=sys.stderr)
    print(f"   Oszczędność: {dead} calls", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
