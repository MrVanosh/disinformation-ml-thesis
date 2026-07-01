"""Generator MANIFEST.md dla datasets/ — SHA-256, rozmiar, źródło, licencja.

Wymóg konkursowy (reprodukowalność): każdy plik źródłowy ma odcisk SHA-256,
metadane pochodzenia, licencję. Plik commit'owany do git (z wyjątkiem datasets/).

Użycie:
    python manifest.py --root datasets/ --output datasets/MANIFEST.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("manifest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Per-dataset metadata (hardcoded, znane źródła)
DATASET_META = {
    "liar": {
        "name": "LIAR",
        "source": "https://huggingface.co/datasets/ucsbnlp/liar",
        "citation": "Wang, W. Y. (2017). 'Liar, Liar Pants on Fire': A New Benchmark Dataset for Fake News Detection.",
        "license": "MIT (UKPLab variant)",
    },
    "truthseeker": {
        "name": "TruthSeeker 2023",
        "source": "https://www.unb.ca/cic/datasets/truthseeker-2023.html",
        "citation": "Dadkhah et al. (2023). The Largest Social Media Ground-Truth Dataset for Real/Fake Content: TruthSeeker.",
        "license": "Research use, CIC UNB",
    },
    "euvsdisinfo": {
        "name": "EUvsDisinfo",
        "source": "https://euvsdisinfo.eu/ + https://data.europa.eu/ + own scraping (trafilatura + DiffBot)",
        "citation": "EU East StratCom Task Force, EEAS. EUvsDisinfo database (2015-2024).",
        "license": "CC-BY-SA 4.0 (debunki); fair use research (scraped article texts)",
    },
    "pl_extra": {
        "name": "Polish disinfo corpus (custom)",
        "source": "Demagog.org.pl (RSS), OKO.press (RSS), Google Fact Check Tools API (CEDMO aggregator), EUvsDisinfo PL subset",
        "citation": "Custom compilation by author for thesis (2026).",
        "license": "Mixed (CC-BY for Demagog/OKO, fair use research for aggregator API). See per-file `license` field.",
    },
    "polygraph": {
        "name": "POLygraph",
        "source": "https://gonito.net/ (planowane, niedostępne w trakcie projektu)",
        "citation": "Mendel et al. POLygraph: Polish disinformation dataset (2024).",
        "license": "Unknown — repo unreachable",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk(root: Path) -> list[tuple[str, Path]]:
    """Zwraca [(dataset_subdir, abs_path)]."""
    out = []
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in sorted(ds_dir.rglob("*")):
            if p.is_file() and not p.name.startswith(".") and p.name != "MANIFEST.md":
                # Skip files > 5 GB (warn)
                size = p.stat().st_size
                if size > 5 * 1024**3:
                    logger.warning("Skipping huge file (>5GB): %s", p)
                    continue
                out.append((ds_dir.name, p))
    return out


def _format_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="datasets/")
    parser.add_argument("--output", default="datasets/MANIFEST.md")
    parser.add_argument("--json-output", default=None,
                        help="Opcjonalnie JSON z pełnymi rekordami (default: <output>.json)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        logger.error("Root not found: %s", root)
        return 1

    files = _walk(root)
    logger.info("Found %d files in %s", len(files), root)

    records: list[dict[str, Any]] = []
    for ds, path in files:
        rel = path.relative_to(root)
        try:
            sha = _sha256(path)
        except Exception as e:
            logger.error("Cannot hash %s: %s", path, e)
            continue
        records.append({
            "dataset": ds,
            "path": str(rel),
            "size_bytes": path.stat().st_size,
            "size_human": _format_size(path.stat().st_size),
            "sha256": sha,
            "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        })

    # Markdown
    lines = [
        "# MANIFEST datasets/",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        f"Total files: {len(records)}",
        f"Total size: {_format_size(sum(r['size_bytes'] for r in records))}",
        "",
    ]

    by_ds: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_ds.setdefault(r["dataset"], []).append(r)

    for ds in sorted(by_ds):
        meta = DATASET_META.get(ds, {})
        lines.append(f"## {meta.get('name', ds)}")
        lines.append("")
        if meta:
            lines.append(f"- **Source**: {meta['source']}")
            lines.append(f"- **Citation**: {meta['citation']}")
            lines.append(f"- **License**: {meta['license']}")
            lines.append("")
        lines.append("| Path | Size | SHA-256 (first 16) | Modified |")
        lines.append("|---|---|---|---|")
        for r in by_ds[ds]:
            lines.append(f"| `{r['path']}` | {r['size_human']} | `{r['sha256'][:16]}…` | {r['mtime'][:10]} |")
        lines.append("")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown manifest → %s", out)

    json_out = Path(args.json_output) if args.json_output else out.with_suffix(".json")
    json_out.write_text(json.dumps({
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "root": str(root),
        "records": records,
        "dataset_meta": DATASET_META,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("JSON manifest → %s", json_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
