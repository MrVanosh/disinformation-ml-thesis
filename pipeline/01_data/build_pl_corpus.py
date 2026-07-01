"""Merge wszystkich źródeł PL + deduplikacja + raport jakości.

Wejście: comma-sep JSONL files (Demagog, OKO, CEDMO, EU PL subset).
Wyjście:
  - JSONL z ujednoliconym schema (gotowy do treningu HerBERT/mBERT).
  - Markdown raport (`pl_corpus_stats.md`) z rozkładem klas, źródeł, długości.

Deduplikacja:
  - Po URL (exact).
  - Near-duplicate via MinHash (Jaccard ≥ threshold).

Opcjonalna augmentacja translation (gdy total < 500):
  - --augment-translation: tłumaczy mBART z RU/EN→PL N wybranych przykładów z EU subset.
    UWAGA: wymaga PyTorch + transformers + mBART model — wolne lokalnie. Domyślnie OFF.

Schema ujednolicony (per row):
  {url, title, text, label, source, language, claim_id?, debunk_id?,
   publication_date?, original_verdict?, license, fetched_at, synthetic_pl?, length}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from datasketch import MinHash, MinHashLSH

logger = logging.getLogger("build_pl_corpus")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.warning("Missing: %s", path)
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _text_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    tokens = (text or "").lower().split()
    for i in range(max(1, len(tokens) - 2)):
        m.update(" ".join(tokens[i:i + 3]).encode("utf-8"))
    return m


def _dedup(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Usuwa duplikaty po URL + near-duplikaty po MinHash."""
    by_url: dict[str, dict[str, Any]] = {}
    for r in rows:
        url = r.get("url", "")
        if not url:
            continue
        if url not in by_url:
            by_url[url] = r
    unique = list(by_url.values())
    logger.info("After URL dedup: %d (from %d)", len(unique), len(rows))

    # Near-dup
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    out_rows: list[dict[str, Any]] = []
    for i, r in enumerate(unique):
        text = r.get("text", "")
        if len(text) < 50:
            continue
        m = _text_minhash(text)
        if lsh.query(m):
            continue  # near-dup of already-accepted
        lsh.insert(f"id_{i}", m)
        out_rows.append(r)
    logger.info("After near-dup (Jaccard ≥ %.2f): %d", threshold, len(out_rows))
    return out_rows


def _normalize_row(r: dict[str, Any]) -> dict[str, Any] | None:
    text = (r.get("text") or "").strip()
    if len(text) < 50:
        return None
    label = r.get("label")
    if label not in (0, 1):
        return None
    # task_type rozróżnia claim-level (fact-check tezy ~ LIAR) od document-level
    # (pełne artykuły ~ EUvsDisinfo). Domyślnie 'claim' (Demagog/OKO), 'article' dla EU subset.
    task_type = r.get("task_type", "claim")
    return {
        "url": r.get("url", ""),
        "title": r.get("title", ""),
        "text": text,
        "label": int(label),
        "task_type": task_type,
        "source": r.get("source", "unknown"),
        "language": r.get("language", "pl"),
        "claim_id": r.get("claim_id"),
        "debunk_id": r.get("debunk_id"),
        "group_key": r.get("claim_id") or r.get("debunk_id") or r.get("url", ""),
        "publication_date": r.get("publication_date") or r.get("date") or r.get("review_date"),
        "original_verdict": r.get("original_verdict"),
        "license": r.get("license", ""),
        "fetched_at": r.get("fetched_at", ""),
        "synthetic_pl": bool(r.get("synthetic_pl", False)),
        "length": len(text),
    }


def _write_stats_report(rows: list[dict[str, Any]], output: Path) -> None:
    counts_by_source = Counter(r["source"] for r in rows)
    counts_by_label = Counter(r["label"] for r in rows)
    lengths = [r["length"] for r in rows]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    med_len = sorted(lengths)[len(lengths) // 2] if lengths else 0

    lines = [
        "# Polski korpus dezinformacji — statystyki",
        "",
        f"- **Total**: {len(rows)}",
        f"- **Średnia długość tekstu**: {avg_len:.0f} znaków",
        f"- **Mediana**: {med_len}",
        "",
        "## Rozkład etykiet",
        "",
        f"| Etykieta | Liczba | % |",
        f"|---|---|---|",
    ]
    for lab in sorted(counts_by_label):
        n = counts_by_label[lab]
        pct = n / len(rows) * 100 if rows else 0
        name = "disinfo" if lab == 1 else "not_disinfo"
        lines.append(f"| {name} ({lab}) | {n} | {pct:.1f}% |")
    lines.extend(["", "## Rozkład źródeł", "", "| Źródło | Liczba | % |", "|---|---|---|"])
    for src, n in counts_by_source.most_common():
        pct = n / len(rows) * 100 if rows else 0
        lines.append(f"| `{src}` | {n} | {pct:.1f}% |")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Stats report → %s", output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, help="Comma-sep JSONL files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--dedup-threshold", type=float, default=0.9)
    parser.add_argument("--augment-translation", action="store_true",
                        help="Augment via mBART RU/EN→PL (slow, may be skipped)")
    parser.add_argument("--augment-count", type=int, default=200)
    parser.add_argument("--min-target", type=int, default=500,
                        help="Min targetowa liczba — jeśli mniej, włącz augmentation (jeśli --augment-translation)")
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    for path_str in args.sources.split(","):
        p = Path(path_str.strip())
        rows = _read_jsonl(p)
        logger.info("Loaded %d from %s", len(rows), p)
        all_rows.extend(rows)

    # Normalize
    normalized = []
    for r in all_rows:
        n = _normalize_row(r)
        if n is not None:
            normalized.append(n)
    logger.info("After normalize: %d", len(normalized))

    # Dedup
    deduped = _dedup(normalized, threshold=args.dedup_threshold)

    # Optional augmentation
    if len(deduped) < args.min_target and args.augment_translation:
        logger.info("Below target (%d < %d) — running translation augmentation",
                    len(deduped), args.min_target)
        try:
            augmented = _augment_translation(deduped, args.augment_count)
            deduped.extend(augmented)
            logger.info("After augmentation: %d", len(deduped))
        except Exception as e:
            logger.error("Augmentation failed: %s — proceeding without", e)

    # Write output — ROZDZIELONE po task_type (dwa pod-zbiory: claims vs articles)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    claims = [r for r in deduped if r.get("task_type") == "claim"]
    articles = [r for r in deduped if r.get("task_type") == "article"]

    # Pliki: <stem>_claims.jsonl i <stem>_articles.jsonl
    stem = out.stem.replace("_claims", "").replace("_articles", "")
    claims_path = out.with_name(f"{stem}_claims.jsonl")
    articles_path = out.with_name(f"{stem}_articles.jsonl")

    def _dump(rows, path):
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote %d rows → %s", len(rows), path)

    if claims:
        _dump(claims, claims_path)
    if articles:
        _dump(articles, articles_path)
    # Zbiorczy (kompatybilność wsteczna)
    _dump(deduped, out)

    # Report — osobno per typ
    _write_stats_report(deduped, Path(args.report))
    logger.info("PL-claims: %d | PL-articles: %d | razem: %d", len(claims), len(articles), len(deduped))

    # Exit code 2 jeśli claims poniżej targetu (sygnał dla automatyki)
    if len(claims) < args.min_target:
        logger.warning("PL-claims %d < min target %d", len(claims), args.min_target)
        return 2
    return 0


def _augment_translation(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Tłumaczenie EN/RU→PL via mBART. Tylko fallback gdy korpus mały."""
    try:
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    except ImportError as e:
        raise RuntimeError(f"mBART requires transformers — {e}")

    logger.info("Loading mBART (~2.5GB)...")
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tok = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    # Wybierz n przykładów z EU subset w innych językach (RU/EN/DE)
    # ten skrypt nie ma dostępu do oryginalnego EU — to placeholder demonstrujący strukturę
    # W praktyce: lokalny CC dostaje osobny --source-for-augment z EU pełnym scrapingiem
    logger.warning("Translation augmentation requires --source-for-augment param (TODO)")
    return []


if __name__ == "__main__":
    sys.exit(main())
