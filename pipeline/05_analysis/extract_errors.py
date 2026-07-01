"""Wyciąga konkretne przykłady błędów (FP/FN) najlepszych modeli — z~tekstem,
mapując indeks predykcji na~wiersz test setu. Do~sekcji 3.5 (taksonomia błędów)."""
import json, sys
from pathlib import Path
sys.path.insert(0, "pipeline/01_data")
from leakage_audit import LOADERS  # noqa


def errors_for(dataset, model, variant, seed=42, n=3):
    ds = LOADERS[dataset](Path("."))
    df = ds.df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    split = json.load(open(f"experiments/splits_v2/{dataset}_seed{seed}.json"))
    test_df = df[df["id"].astype(str).isin(set(split["test_ids"]))].reset_index(drop=True)

    preds = [json.loads(l) for l in open(f"experiments/preds_v2/{dataset}_{model}_{variant}_seed{seed}.jsonl")]
    # sanity: czy y_true z preds zgadza się z test_df labels (sprawdza mapowanie)
    match = sum(1 for k, p in enumerate(preds) if k < len(test_df)
                and int(p["y_true"]) == int(test_df.iloc[k]["label"]))
    print(f"\n### {dataset}/{model}/{variant}: mapowanie i->tekst {match}/{len(preds)} zgodne "
          f"({'OK' if match/len(preds) > 0.95 else 'UWAGA niespójne'})")

    fp, fn = [], []
    for k, p in enumerate(preds):
        if k >= len(test_df):
            break
        yt, yp = int(p["y_true"]), int(p["y_pred"])
        if yt == yp:
            continue
        txt = str(test_df.iloc[k]["text"])[:200].replace("\n", " ")
        if yt == 0 and yp == 1 and len(fp) < n:  # FP: prawda oznaczona jako dezinfo
            fp.append(txt)
        elif yt == 1 and yp == 0 and len(fn) < n:  # FN: dezinfo oznaczona jako prawda
            fn.append(txt)
    print("  FP (prawda→dezinfo):")
    for t in fp:
        print(f"    • {t}")
    print("  FN (dezinfo→prawda):")
    for t in fn:
        print(f"    • {t}")


if __name__ == "__main__":
    errors_for("liar", "lr", "tfidf")
    errors_for("truthseeker", "lr", "tfidf")
    errors_for("pl_claims", "lr", "tfidf")
