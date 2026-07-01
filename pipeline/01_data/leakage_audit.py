"""Multi-level leakage audit dla benchmarków detekcji dezinformacji.

Sprawdza pięć poziomów wycieku danych między train a test split:
  1. Group-level overlap (statement_clean dla TS, debunk_id dla EU).
  2. Temporal leakage (test_dates ⊂ train_date_range).
  3. Source domain leakage (publishing domain in EU).
  4. Near-duplicate leakage (MinHash text similarity ≥ 0.85).
  5. Label consistency within group (czy każda grupa ma jedną etykietę).

Wynik: markdown raport + JSON ze szczegółami + opcjonalnie heatmap PNG.

Użycie:
    python leakage_audit.py --dataset truthseeker \\
        --output pipeline/06_thesis_inputs/audit/truthseeker_initial.md

Skrypt jest deterministyczny przy ustalonym seed (--seed, default 42).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import GroupShuffleSplit, train_test_split

logger = logging.getLogger("leakage_audit")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ────────────────────────────────────────────────────────────────────────────
# Dataset loaders — adaptery do obecnej struktury katalogów
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class AuditDataset:
    """Znormalizowana reprezentacja zbioru dla audytu."""

    name: str
    df: pd.DataFrame  # kolumny: id, text, label, group_key, date (opt.), source_url (opt.)
    group_column: str  # nazwa kolumny grupującej (np. statement_clean, debunk_id)
    has_temporal: bool = False
    has_source: bool = False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.warning("File not found: %s", path)
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Bad JSON line in %s: %s", path, exc)
    return rows


def load_liar(root: Path) -> AuditDataset:
    """LIAR: oficjalne UKPLab splits. Group = statement_id (każde stwierdzenie unikalne).

    Lokalna struktura: liar_{train,validation,test}.csv z kolumnami
    text, label_text, labels, context (wariant binarny UKPLab/liar).
    """
    csv_dir = root / "datasets" / "liar"
    csv_parts = []
    for split in ("train", "validation", "test"):
        p = csv_dir / f"liar_{split}.csv"
        if p.exists():
            part = pd.read_csv(p)
            part["_split"] = split
            csv_parts.append(part)
    if csv_parts:
        df = pd.concat(csv_parts, ignore_index=True)
        # Binarny wariant: label_text "false statement" → 1 (disinfo), "true statement" → 0
        if "label_text" in df.columns:
            df["label"] = (df["label_text"].astype(str).str.contains("false", case=False)).astype(int)
        elif "labels" in df.columns:
            df["label"] = (pd.to_numeric(df["labels"], errors="coerce") < 3).astype(int)
        df["text"] = df.get("text", "").astype(str)
        df["id"] = df.index.astype(str)
    else:
        # Fallback: jsonl lub HuggingFace
        candidates = [
            csv_dir / "binary.jsonl", csv_dir / "all.jsonl", csv_dir / "combined.jsonl",
        ]
        df = None
        for p in candidates:
            if p.exists():
                df = pd.DataFrame(_read_jsonl(p))
                break
        if df is None:
            try:
                from datasets import load_dataset
                ds = load_dataset("ucsbnlp/liar")
                df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
                df["text"] = df.get("statement", df.get("text", ""))
                df["label"] = (df["label"] >= 3).astype(int)
                df["id"] = df.index.astype(str)
            except Exception as e:
                raise RuntimeError(f"Cannot load LIAR from {csv_dir} or HF: {e}")

    df["group_key"] = df["id"].astype(str)  # każdy statement własna grupa
    return AuditDataset(name="liar", df=df, group_column="group_key", has_temporal=False)


def load_truthseeker(root: Path) -> AuditDataset:
    """TruthSeeker 2023: 134k tweetów odnoszących się do statementów PolitiFact."""
    candidates = [
        root / "datasets" / "truthseeker" / "Features_For_Traditional_ML_Techniques.csv",
        root / "datasets" / "truthseeker" / "truthseeker.csv",
        root / "datasets" / "truthseeker" / "truthseeker_train.csv",
    ]
    df = None
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            break
    if df is None:
        raise RuntimeError(f"Cannot load TruthSeeker from {candidates}")

    # Normalizacja kolumn
    text_col = next((c for c in ("tweet", "statement", "text") if c in df.columns), None)
    label_col = next((c for c in ("majority_target", "BinaryNumTarget", "label") if c in df.columns), None)
    if text_col is None or label_col is None:
        raise RuntimeError(f"TS columns missing — got {df.columns.tolist()}")

    df = df.rename(columns={text_col: "text", label_col: "label"})
    # fillna→str: TS ma pojedyncze wiersze z NaN w text; object+astype(str) NIE konwertuje
    # np.nan, więc tokenizer/wektoryzator dostaje float nan i wywala się. Czyścimy globalnie.
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].map(lambda x: 1 if str(x).lower() in ("true", "1", "1.0") else 0)
    df["id"] = df.get("tweet_id", df.index).astype(str)

    # Group key = statement (znormalizowany)
    if "statement" in df.columns:
        df["group_key"] = df["statement"].astype(str).map(_normalize_statement)
    elif "statement_clean" in df.columns:
        df["group_key"] = df["statement_clean"].astype(str)
    else:
        logger.warning("TS: no statement column — using text as group_key (high leakage if duplicates)")
        df["group_key"] = df["text"].astype(str).map(_normalize_statement)

    has_temporal = "created_at" in df.columns or "tweet_date" in df.columns
    if has_temporal:
        date_col = "created_at" if "created_at" in df.columns else "tweet_date"
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")

    return AuditDataset(
        name="truthseeker",
        df=df,
        group_column="group_key",
        has_temporal=has_temporal,
    )


def load_euvsdisinfo(root: Path) -> AuditDataset:
    """EUvsDisinfo: artykuły powiązane z debunkami, ze scrapingu (trafilatura + DiffBot)."""
    sources = [
        root / "datasets" / "euvsdisinfo" / "scraped.jsonl",
        root / "datasets" / "euvsdisinfo" / "scraped_diffbot.jsonl",
    ]
    rows: list[dict[str, Any]] = []
    for p in sources:
        rows.extend(_read_jsonl(p))

    if not rows:
        raise RuntimeError(f"EU empty — no data in {sources}")

    df = pd.DataFrame(rows)

    # Deduplikacja po article_url
    if "article_url" in df.columns:
        df = df.drop_duplicates(subset=["article_url"], keep="first").reset_index(drop=True)

    # Mapowanie pól lokalnej struktury scraped.jsonl:
    #   article_text → text, class (disinformation/trustworthy) → label (1/0)
    if "article_text" in df.columns:
        df["text"] = df["article_text"].fillna("").astype(str)
    elif "text" in df.columns:
        df["text"] = df["text"].fillna("").astype(str)
    else:
        df["text"] = df.get("content", "")

    if "class" in df.columns:
        df["label"] = (df["class"].astype(str).str.lower() == "disinformation").astype(int)
    elif "label" in df.columns:
        df["label"] = df["label"].astype(int)
    elif "is_disinfo" in df.columns:
        df["label"] = df["is_disinfo"].astype(int)
    else:
        df["label"] = 1
    df["id"] = df.get("article_url", df.index).astype(str)

    # Język artykułu (przyda się dla audytu i PL subset)
    if "article_language" in df.columns:
        df["lang"] = df["article_language"].astype(str)

    # Group key = debunk_id (jeden debunk → wiele artykułów źródłowych)
    if "debunk_id" in df.columns:
        df["group_key"] = df["debunk_id"].astype(str)
    elif "case_id" in df.columns:
        df["group_key"] = df["case_id"].astype(str)
    else:
        logger.warning("EU: no debunk_id — using article_url as group_key (no grouping)")
        df["group_key"] = df["id"]

    # Temporal
    date_candidates = ["publication_date", "debunk_date", "date"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    has_temporal = date_col is not None
    if has_temporal:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")

    # Source domain
    has_source = "article_url" in df.columns
    if has_source:
        df["source_domain"] = df["article_url"].map(_extract_domain)

    return AuditDataset(
        name="euvsdisinfo",
        df=df,
        group_column="group_key",
        has_temporal=has_temporal,
        has_source=has_source,
    )


def _load_pl_variant(root: Path, variant: str) -> AuditDataset:
    """Polski korpus — wariant 'claims' (Demagog/OKO claimReviewed ~ LIAR)
    lub 'articles' (EU PL subset, pełne artykuły ~ EUvsDisinfo)."""
    fname = f"corpus_pl_{variant}.jsonl"
    path = root / "datasets" / "pl_extra" / fname
    rows = _read_jsonl(path)
    if not rows:
        # fallback: zbiorczy plik
        rows = _read_jsonl(root / "datasets" / "pl_extra" / "corpus_pl.jsonl")
        rows = [r for r in rows if r.get("task_type", "claim") == ("claim" if variant == "claims" else "article")]
    if not rows:
        raise RuntimeError(f"PL {variant} empty — uruchom build_pl_corpus.py ({path})")

    df = pd.DataFrame(rows)
    df["text"] = df.get("text", "").astype(str)
    df["label"] = df.get("label", 0).astype(int)
    df["id"] = df.get("url", df.index).astype(str)
    if "group_key" in df.columns:
        df["group_key"] = df["group_key"].astype(str)
    elif "claim_id" in df.columns:
        df["group_key"] = df["claim_id"].astype(str)
    elif "debunk_id" in df.columns:
        df["group_key"] = df["debunk_id"].astype(str)
    else:
        df["group_key"] = df["id"]

    has_temporal = "publication_date" in df.columns or "date" in df.columns
    if has_temporal:
        date_col = "publication_date" if "publication_date" in df.columns else "date"
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    has_source = "url" in df.columns and df["url"].astype(bool).any()
    if has_source:
        df["source_domain"] = df["url"].map(_extract_domain)

    return AuditDataset(name=f"pl_{variant}", df=df, group_column="group_key",
                        has_temporal=has_temporal, has_source=has_source)


def load_pl_claims(root: Path) -> AuditDataset:
    return _load_pl_variant(root, "claims")


def load_pl_articles(root: Path) -> AuditDataset:
    return _load_pl_variant(root, "articles")


def load_pl_corpus(root: Path) -> AuditDataset:
    """Wstecznie kompatybilny alias — zwraca claims (główny polski pod-zbiór)."""
    return load_pl_claims(root)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _normalize_statement(s: str) -> str:
    """Lower + remove punctuation + collapse whitespace. Używane jako group key dla TS."""
    if not isinstance(s, str):
        # NaN/float/None — zwróć pusty string (zostanie własną grupą)
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        # Remove www.
        if host.startswith("www."):
            host = host[4:]
        return host.lower()
    except Exception:
        return ""


def _text_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    # Shingle 3-word
    tokens = (text or "").lower().split()
    for i in range(max(1, len(tokens) - 2)):
        shingle = " ".join(tokens[i:i + 3]).encode("utf-8")
        m.update(shingle)
    return m


# ────────────────────────────────────────────────────────────────────────────
# Audit functions
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class AuditReport:
    name: str
    n_total: int
    n_train: int
    n_test: int
    overlaps: dict[str, dict[str, Any]] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# Audyt wycieku danych — {self.name}")
        lines.append("")
        lines.append(f"- **N total**: {self.n_total:,}")
        lines.append(f"- **N train**: {self.n_train:,}")
        lines.append(f"- **N test**: {self.n_test:,}")
        lines.append("")
        lines.append("## Wyniki audytu (5 poziomów)")
        lines.append("")
        lines.append("| Poziom | Status | Metryka | Wartość |")
        lines.append("|---|---|---|---|")
        for key, info in self.overlaps.items():
            status = info.get("status", "?")
            metric = info.get("metric", "")
            value = info.get("value", "")
            lines.append(f"| {key} | {status} | {metric} | {value} |")
        lines.append("")
        if self.recommendations:
            lines.append("## Rekomendacje")
            lines.append("")
            for r in self.recommendations:
                lines.append(f"- {r}")
            lines.append("")
        return "\n".join(lines)


def audit_group_overlap(
    ds: AuditDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    report: AuditReport,
) -> None:
    """Sprawdza ile grup z testu pojawia się też w treningu (każde >0 = leakage)."""
    train_groups = set(ds.df.loc[train_idx, ds.group_column].tolist())
    test_groups = set(ds.df.loc[test_idx, ds.group_column].tolist())
    overlap = train_groups & test_groups
    n_test_groups = len(test_groups)
    pct = (len(overlap) / n_test_groups * 100) if n_test_groups else 0.0
    status = "OK" if pct == 0 else ("WYCIEK" if pct > 5 else "ostrzeżenie")
    report.overlaps["1. Group overlap"] = {
        "status": status,
        "metric": f"% grup testowych obecnych też w treningu",
        "value": f"{pct:.2f}% ({len(overlap)}/{n_test_groups})",
    }
    if pct > 0:
        report.recommendations.append(
            f"Zastosuj `GroupShuffleSplit` po kolumnie `{ds.group_column}` — obecny split zawiera "
            f"{len(overlap)} grup nakładających się."
        )


def audit_temporal(
    ds: AuditDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    report: AuditReport,
) -> None:
    if not ds.has_temporal or "date" not in ds.df.columns:
        report.overlaps["2. Temporal leakage"] = {
            "status": "N/A",
            "metric": "brak daty publikacji",
            "value": "-",
        }
        return
    train_dates = ds.df.loc[train_idx, "date"].dropna()
    test_dates = ds.df.loc[test_idx, "date"].dropna()
    if train_dates.empty or test_dates.empty:
        report.overlaps["2. Temporal leakage"] = {
            "status": "N/A", "metric": "brak prawidłowych dat", "value": "-"
        }
        return
    train_max = train_dates.max()
    test_min = test_dates.min()
    leak_pct = (test_dates < train_max).mean() * 100
    status = "OK" if leak_pct < 5 else ("ostrzeżenie" if leak_pct < 50 else "uwaga")
    report.overlaps["2. Temporal leakage"] = {
        "status": status,
        "metric": "% próbek testowych z datą < max(train)",
        "value": f"{leak_pct:.1f}% (train_max={train_max:%Y-%m-%d}, test_min={test_min:%Y-%m-%d})",
    }
    if leak_pct > 50:
        report.recommendations.append(
            "Większość testowych próbek ma datę wcześniejszą niż max(train) — "
            "rozważ `TimeSeriesSplit` lub jawne odcięcie po dacie."
        )


def audit_source_domain(
    ds: AuditDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    report: AuditReport,
) -> None:
    if not ds.has_source or "source_domain" not in ds.df.columns:
        report.overlaps["3. Source domain leakage"] = {
            "status": "N/A", "metric": "brak URL", "value": "-"
        }
        return
    train_domains = ds.df.loc[train_idx, "source_domain"].dropna()
    test_domains = ds.df.loc[test_idx, "source_domain"].dropna()
    common = set(train_domains) & set(test_domains)
    n_test_domains = test_domains.nunique()
    pct_test_in_train = (test_domains.isin(common)).mean() * 100
    status = "wysoki" if pct_test_in_train > 80 else "umiarkowany"
    report.overlaps["3. Source domain leakage"] = {
        "status": status,
        "metric": "% próbek testowych z domeny obecnej w trainie",
        "value": f"{pct_test_in_train:.1f}% (wspólnych domen: {len(common)}/{n_test_domains})",
    }
    if pct_test_in_train > 90:
        report.recommendations.append(
            "Niemal wszystkie domeny testowe są obecne w trainie — model może "
            "uczyć się 'rozpoznawania domeny' zamiast 'rozpoznawania dezinformacji'. "
            "Rozważ split po domenach (DomainAwareSplit)."
        )


def audit_near_duplicates(
    ds: AuditDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    report: AuditReport,
    threshold: float = 0.85,
    sample_size: int = 5000,
) -> None:
    """MinHash LSH: znajduje pary train↔test z Jaccard ≥ threshold."""
    # Sample dla performance (LSH na 134k tweetów byłoby wolne)
    rng = np.random.default_rng(42)
    train_sample_idx = rng.choice(train_idx, size=min(sample_size, len(train_idx)), replace=False)
    test_sample_idx = rng.choice(test_idx, size=min(sample_size, len(test_idx)), replace=False)

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    train_mhs: dict[str, MinHash] = {}
    for i in train_sample_idx:
        text = str(ds.df.at[i, "text"])
        if len(text) < 20:
            continue
        m = _text_minhash(text)
        key = f"tr_{i}"
        lsh.insert(key, m)
        train_mhs[key] = m

    near_dup_count = 0
    for i in test_sample_idx:
        text = str(ds.df.at[i, "text"])
        if len(text) < 20:
            continue
        m = _text_minhash(text)
        matches = lsh.query(m)
        if matches:
            near_dup_count += 1

    pct = (near_dup_count / max(1, len(test_sample_idx))) * 100
    extrapolated = pct  # próbka reprezentatywna → ekstrapolacja
    status = "OK" if pct < 1 else ("ostrzeżenie" if pct < 5 else "WYCIEK")
    report.overlaps["4. Near-duplicate leakage"] = {
        "status": status,
        "metric": f"% próbek testowych z MinHash sim ≥ {threshold} do dowolnej w train (sample {sample_size})",
        "value": f"{pct:.2f}% (~{extrapolated:.1f}% ekstrap.)",
    }
    if pct > 5:
        report.recommendations.append(
            f"≥5% próbek testowych ma near-duplicate w trainie (Jaccard ≥ {threshold}). "
            "Wykonaj dedup MinHash przed splittingiem."
        )


def audit_label_consistency(ds: AuditDataset, report: AuditReport) -> None:
    """W grupach (group_key) — czy etykiety są spójne?"""
    grp = ds.df.groupby(ds.group_column)["label"].agg(["nunique", "count"])
    inconsistent = grp[grp["nunique"] > 1]
    n_groups = len(grp)
    n_inconsistent = len(inconsistent)
    pct = (n_inconsistent / n_groups * 100) if n_groups else 0.0
    status = "OK" if pct < 1 else ("uwaga" if pct < 10 else "problem")
    report.overlaps["5. Label consistency in group"] = {
        "status": status,
        "metric": f"% grup z więcej niż 1 unikalną etykietą",
        "value": f"{pct:.2f}% ({n_inconsistent}/{n_groups})",
    }
    if pct > 5:
        report.recommendations.append(
            f"{n_inconsistent} grup ma mieszane etykiety — sprawdź czy stosujemy 'majority label per group' "
            "lub czy faktycznie istnieją sprzeczności w danych źródłowych."
        )


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


LOADERS = {
    "liar": load_liar,
    "truthseeker": load_truthseeker,
    "euvsdisinfo": load_euvsdisinfo,
    "pl_corpus": load_pl_corpus,    # alias = pl_claims (wsteczna kompatybilność)
    "pl_claims": load_pl_claims,
    "pl_articles": load_pl_articles,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(LOADERS.keys()))
    parser.add_argument("--root", default=".", help="Root katalogu projektu (zawiera datasets/)")
    parser.add_argument("--output", required=True, help="Plik wyjściowy markdown")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--split-mode", choices=["random", "grouped"], default="random",
                        help="Audytujemy random (default — diagnostyka) lub już grouped split")
    parser.add_argument("--minhash-sample", type=int, default=5000)
    parser.add_argument("--minhash-threshold", type=float, default=0.85)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    loader = LOADERS[args.dataset]
    ds = loader(root)
    logger.info("Loaded %s: %d rows, group_column=%s", ds.name, len(ds.df), ds.group_column)

    # Generuj split (dla diagnostyki)
    if args.split_mode == "random":
        train_idx, test_idx = train_test_split(
            np.arange(len(ds.df)), test_size=args.test_size, random_state=args.seed,
            stratify=ds.df["label"] if ds.df["label"].nunique() > 1 else None,
        )
    else:  # grouped
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(ds.df, groups=ds.df[ds.group_column]))

    # Reset index aby pasował do .loc
    ds.df = ds.df.reset_index(drop=True)
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    report = AuditReport(
        name=f"{ds.name} ({args.split_mode} split, seed={args.seed})",
        n_total=len(ds.df),
        n_train=len(train_idx),
        n_test=len(test_idx),
    )

    audit_group_overlap(ds, train_idx, test_idx, report)
    audit_temporal(ds, train_idx, test_idx, report)
    audit_source_domain(ds, train_idx, test_idx, report)
    audit_near_duplicates(ds, train_idx, test_idx, report, threshold=args.minhash_threshold,
                          sample_size=args.minhash_sample)
    audit_label_consistency(ds, report)

    # Zapisz markdown + JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.to_markdown(), encoding="utf-8")
    logger.info("Markdown report → %s", output_path)

    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "name": report.name,
        "n_total": report.n_total,
        "n_train": report.n_train,
        "n_test": report.n_test,
        "overlaps": report.overlaps,
        "recommendations": report.recommendations,
        "split_config": {"mode": args.split_mode, "test_size": args.test_size, "seed": args.seed},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("JSON details → %s", json_path)

    # Print summary do stdout
    print("\n" + "=" * 70)
    print(report.to_markdown())

    return 0


if __name__ == "__main__":
    sys.exit(main())
