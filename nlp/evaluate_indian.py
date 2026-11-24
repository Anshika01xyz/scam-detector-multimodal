"""Step 6: compare TF-IDF and DistilBERT on the unseen Indian-style test messages.

Uses the true message text from data/screenshots/labels.csv (no OCR involved),
so this measures the text models alone. Run after both models are trained:
    python ocr/make_synthetic_screenshots.py     # creates labels.csv if missing
    python nlp/evaluate_indian.py

Saves reports/indian_eval.json
"""
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean import clean_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "screenshots" / "labels.csv"
TFIDF = ROOT / "models" / "tfidf_lr.pkl"
BERT_DIR = ROOT / "models" / "distilbert_scam"
REPORTS = ROOT / "reports"


def score_tfidf(texts):
    model = joblib.load(TFIDF)
    return model.predict_proba([clean_text(t) for t in texts])[:, 1]


def score_distilbert(texts):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from train_distilbert import predict_proba
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(BERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(device)
    return predict_proba(model, tok, texts, device)


def summarize(name, y, probs, df):
    pred = (probs >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    print(f"\n=== {name} ===")
    print(f"accuracy {np.mean(pred == y):.3f} | scam precision {p:.3f} recall {r:.3f} F1 {f1:.3f}")
    print(f"legit wrongly flagged: {cm[0][1]}/{cm[0].sum()} | scams missed: {cm[1][0]}/{cm[1].sum()}")
    wrong = df[pred != y]
    for idx, row in wrong.iterrows():
        print(f"  WRONG ({'scam' if row.label else 'legit'}, got {probs[idx]:.2f}) {row.text[:75]}")
    return {"precision": p, "recall": r, "f1": f1, "confusion_matrix": cm.tolist(),
            "accuracy": float(np.mean(pred == y))}


def main():
    if not LABELS.exists():
        print("labels.csv missing, generating screenshots first...")
        subprocess.run([sys.executable, str(ROOT / "ocr" / "make_synthetic_screenshots.py")], check=True)
    df = pd.read_csv(LABELS)
    y = df["label"].to_numpy()
    print(f"{len(df)} messages ({y.sum()} scam, {len(y) - y.sum()} legit)")

    results = {}
    if TFIDF.exists():
        results["tfidf"] = summarize("TF-IDF + LR", y, score_tfidf(df["text"].tolist()), df)
    else:
        print(f"\n[skip] {TFIDF} not found. Run nlp/train_tfidf.py first.")
    if BERT_DIR.exists():
        results["distilbert"] = summarize("DistilBERT", y, score_distilbert(df["text"].tolist()), df)
    else:
        print(f"\n[skip] {BERT_DIR} not found. Run nlp/train_distilbert.py first.")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "indian_eval.json").write_text(json.dumps(results, indent=2))
    print("\nSaved reports/indian_eval.json")


if __name__ == "__main__":
    main()
