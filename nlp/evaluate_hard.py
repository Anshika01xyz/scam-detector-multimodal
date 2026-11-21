"""Evaluate TF-IDF and DistilBERT on the hard test set (data/test/hard_test.csv).

    python nlp/evaluate_hard.py

Prints: overlap check against the training data, overall metrics with a 95% confidence
interval, a per-category accuracy table, and every mistake. Saves reports/hard_eval.json.

IMPORTANT: this set is for testing only. Do not train on it or tune settings on it
(if you do, the numbers stop being honest).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean import clean_text  # noqa: E402
from evaluate_indian import BERT_DIR, TFIDF, score_distilbert, score_tfidf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def wilson(k, n, z=1.96):
    """95% confidence interval for an accuracy of k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def overlap_check(df):
    train = ROOT / "data" / "processed"
    if not (train / "train.csv").exists():
        print("[skip] overlap check: run nlp/prepare_data.py first\n")
        return
    ref = pd.concat([pd.read_csv(train / "train.csv"), pd.read_csv(train / "test.csv")])["clean_text"].tolist()
    mine = [clean_text(t) for t in df["text"]]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(ref + mine)
    sims = (vec.transform(mine) @ vec.transform(ref).T).max(axis=1).toarray().ravel()
    print(f"Overlap with training data (char n-gram cosine similarity): mean {sims.mean():.2f}, max {sims.max():.2f}")
    near = int((sims > 0.8).sum())
    print(f"messages with similarity > 0.8 to a training message: {near}/{len(df)}")
    for i in np.argsort(-sims)[:min(near, 5)]:
        print(f"   {sims[i]:.2f}  {df.text[i][:80]}")
    print()


def evaluate(name, probs, df):
    y = df["label"].to_numpy()
    pred = (probs >= 0.5).astype(int)
    correct = int((pred == y).sum())
    lo, hi = wilson(correct, len(y))
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    print(f"=== {name} ===")
    print(f"accuracy {correct / len(y):.3f}  (95% CI {lo:.3f} to {hi:.3f}, n={len(y)})")
    print(f"scam precision {p:.3f} | recall {r:.3f} | F1 {f1:.3f}")
    print(f"legit wrongly flagged: {cm[0][1]}/{cm[0].sum()} | scams missed: {cm[1][0]}/{cm[1].sum()}\n")
    by_cat = {c: float((pred[g.index] == y[g.index]).mean()) for c, g in df.groupby("category")}
    return {"accuracy": correct / len(y), "ci95": [lo, hi], "precision": p, "recall": r, "f1": f1,
            "confusion_matrix": cm.tolist(), "by_category": by_cat}, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "test" / "hard_test.csv"))
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    print(f"{len(df)} messages ({int(df.label.sum())} scam, {int((1 - df.label).sum())} legit)\n")
    overlap_check(df)

    texts = df["text"].tolist()
    results, preds, probs_all = {}, {}, {}
    if TFIDF.exists():
        probs_all["TF-IDF"] = score_tfidf(texts)
    else:
        print(f"[skip] {TFIDF} not found\n")
    if BERT_DIR.exists():
        probs_all["DistilBERT"] = score_distilbert(texts)
    else:
        print(f"[skip] {BERT_DIR} not found\n")
    for name, probs in probs_all.items():
        results[name], preds[name] = evaluate(name, probs, df)

    if results:
        cats = sorted(df.category.unique(), key=lambda c: (c.startswith("legit"), c))
        print("Accuracy by category" + " " * 8 + "".join(f"{n:>12s}" for n in results) + "     n")
        for c in cats:
            n = int((df.category == c).sum())
            print(f"{c:28s}" + "".join(f"{results[m]['by_category'][c]:12.2f}" for m in results) + f"{n:6d}")
        for name in results:
            wrong = df[preds[name] != df.label.to_numpy()]
            print(f"\n--- {name}: {len(wrong)} mistakes ---")
            for idx, row in wrong.iterrows():
                print(f"  [{row.category}] true {'scam ' if row.label else 'legit'} got {probs_all[name][idx]:.2f} | {row.text[:80]}")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "hard_eval.json").write_text(json.dumps(results, indent=2))
    print("\nSaved reports/hard_eval.json")


if __name__ == "__main__":
    main()
