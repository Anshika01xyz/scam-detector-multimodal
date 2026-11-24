"""Full pipeline test for the course models: screenshot -> OCR -> classifier -> scam / legit.

    python nlp/evaluate_pipeline_classical.py
    python nlp/evaluate_pipeline_classical.py --engine easyocr --refresh

For each screenshot tier (data/screenshots = clean, data/screenshots_messy = messy) it
  1. reads the text with OCR (cached in reports/ocr_text_<tier>.csv; use --refresh to redo),
  2. trains every model from nlp/train_classical.py (same features and data),
  3. classifies (a) the TRUE text from labels.csv and (b) the OCR text.
Comparing (a) with (b) shows how much OCR mistakes hurt each model.
Saves reports/pipeline_comparison.csv.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nlp"))
sys.path.insert(0, str(ROOT / "ocr"))
from clean import clean_text  # noqa: E402
from extract import extract_text  # noqa: E402
from train_classical import INDIA_REPEAT, PROC, metrics, make_models  # noqa: E402

TIERS = {"clean": "data/screenshots", "messy": "data/screenshots_messy"}
REPORTS = ROOT / "reports"


def ocr_tier(tier, folder, engine, refresh):
    cache = REPORTS / f"ocr_text_{tier}.csv"
    labels = pd.read_csv(ROOT / folder / "labels.csv")
    if cache.exists() and not refresh:
        return pd.read_csv(cache).fillna({"ocr_text": ""})
    texts = []
    for name in labels["filename"]:
        texts.append(extract_text(ROOT / folder / name, mode="auto", engine=engine)["text"])
        print(f"  OCR {tier}: {len(texts)}/{len(labels)}", end="\r")
    labels["ocr_text"] = texts
    REPORTS.mkdir(exist_ok=True)
    labels.to_csv(cache, index=False)
    print()
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="auto", choices=["auto", "easyocr", "tesseract"])
    ap.add_argument("--refresh", action="store_true", help="redo OCR even if cached")
    args = ap.parse_args()

    data = {t: ocr_tier(t, f, args.engine, args.refresh) for t, f in TIERS.items()}

    train = pd.read_csv(PROC / "train.csv")
    india = train[train["source"] == "my_scams"]
    train_aug = pd.concat([train] + [india] * (INDIA_REPEAT - 1), ignore_index=True)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_tr = vec.fit_transform(train_aug["clean_text"])
    y_tr = train_aug["label"].to_numpy()
    n_pos = int(y_tr.sum())

    rows = []
    for name, model in make_models(len(y_tr) - n_pos, n_pos).items():
        dense = name == "AdaBoost"
        model.fit(X_tr.toarray() if dense else X_tr, y_tr)
        for tier, df in data.items():
            for source, col in (("true text", "text"), ("OCR text", "ocr_text")):
                X = vec.transform(df[col].apply(clean_text))
                pred = model.predict(X.toarray() if dense else X)
                rows.append({"model": name, "tier": tier, "input": source,
                             **metrics(df["label"].to_numpy(), pred)})
        print(f"done: {name}")

    out = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    out.to_csv(REPORTS / "pipeline_comparison.csv", index=False)
    for tier in TIERS:
        d = out[out["tier"] == tier]
        piv = d.pivot(index="model", columns="input", values=["accuracy", "recall", "specificity"])
        print(f"\n=== {tier.upper()} screenshots ({len(data[tier])} images) ===")
        print(piv.round(3).to_string())
    print(f"\nSaved {REPORTS / 'pipeline_comparison.csv'}")


if __name__ == "__main__":
    main()
