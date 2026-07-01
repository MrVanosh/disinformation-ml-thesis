"""Wyciągnij polskie artykuły ze scrapingu EUvsDisinfo (trafilatura + DiffBot).

Wejście: dwa pliki JSONL (lub więcej, comma-sep).
Wyjście: polskie artykuły w formacie zgodnym z PL corpus (uniform schema).

Filtr języka:
  - Pole 'lang' lub 'language' (z trafilatura/DiffBot).
  - Fallback: langdetect na text (jeśli pakiet zainstalowany).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("pl_eu_extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _is_polish(row: dict[str, Any]) -> bool:
    lang = (row.get("lang") or row.get("language") or "").lower()
    if lang in ("pl", "polish", "polski"):
        return True
    if lang and lang not in ("pl",):
        return False  # already classified as non-PL
    # Fallback: langdetect
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        text = (row.get("text") or "")[:500]
        if len(text) < 50:
            return False
        return detect(text) == "pl"
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Comma-sep list of JSONL files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    inputs = [Path(p.strip()) for p in args.input.split(",")]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_pl = 0

    with out.open("w", encoding="utf-8") as f_out:
        for inp in inputs:
            if not inp.exists():
                logger.warning("Missing %s", inp)
                continue
            with inp.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    n_total += 1
                    if not _is_polish(row):
                        continue
                    text = (row.get("text") or "").strip()
                    if len(text) < 100:
                        continue
                    normalized = {
                        "url": row.get("article_url") or row.get("url"),
                        "title": row.get("title", ""),
                        "text": text,
                        "label": int(row.get("label", 1)),  # EU = disinfo bias
                        "debunk_id": row.get("debunk_id") or row.get("case_id"),
                        "publication_date": row.get("publication_date") or row.get("date"),
                        "source": "euvsdisinfo_pl_subset",
                        "language": "pl",
                        "scraper": row.get("scraper", "trafilatura"),
                        "fetched_at": datetime.utcnow().isoformat(),
                        "license": "CC-BY-SA-4.0 (EUvsDisinfo)",
                    }
                    f_out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    n_pl += 1

    logger.info("Scanned %d rows, extracted %d Polish", n_total, n_pl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
