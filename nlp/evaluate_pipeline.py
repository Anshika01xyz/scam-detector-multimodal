"""End-to-end test: screenshot -> OCR -> text model -> verdict, on data/screenshots.

    python nlp/evaluate_pipeline.py                 # model auto, OCR auto, binary preprocessing
    python nlp/evaluate_pipeline.py --model tfidf --ocr-mode gray

Saves reports/pipeline_eval.json. A screenshot counts as "flagged" if text_score >= 0.5.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nlp"))
from predict import analyze, resolve_model  # noqa: E402

SHOTS = ROOT / "data" / "screenshots"
TYPE_MAP = {"kyc": "Bank / KYC fraud", "upi": "UPI / payment scam", "lottery": "Fake lottery / prize",
            "job": "Fake job / task offer", "electricity": "Impersonation / urgency",
            "parcel": "Impersonation / urgency", "other": "Impersonation / urgency",
            "loan": "Other / unknown"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto", choices=["auto", "tfidf", "distilbert"])
    ap.add_argument("--ocr-engine", default="auto", choices=["auto", "easyocr", "tesseract"])
    ap.add_argument("--ocr-mode", default="binary", choices=["none", "gray", "binary"])
    a = ap.parse_args()

    df = pd.read_csv(SHOTS / "labels.csv")
    rows = []
    for _, r in df.iterrows():
        res = analyze(image=SHOTS / r["filename"], model=a.model, ocr_engine=a.ocr_engine, ocr_mode=a.ocr_mode)
        score = res["text_score"] if res["text_score"] is not None else 0.0
        rows.append({"file": r["filename"], "label": int(r["label"]), "score": score,
                     "verdict": res["text_verdict"], "type": res["scam_type"],
                     "true_type": TYPE_MAP.get(r["scam_type"]), "ocr_conf": res["ocr_confidence"]})
    out = pd.DataFrame(rows)
    pred = (out.score >= 0.5).astype(int)
    scams = out[out.label == 1]
    print(f"Model: {resolve_model(a.model)} | OCR mode: {a.ocr_mode} | {len(out)} screenshots")
    print(f"accuracy {np.mean(pred == out.label):.3f} | scams caught {int(pred[out.label == 1].sum())}/{len(scams)}"
          f" | legit wrongly flagged {int(pred[out.label == 0].sum())}/{int((out.label == 0).sum())}")
    print(f"verdicts: {out.verdict.value_counts(dropna=False).to_dict()}")
    print(f"scam type correct (among scams caught): "
          f"{int((scams.type == scams.true_type).sum())}/{int((scams.verdict != 'legit').sum())}")
    print(f"mean OCR confidence: {out.ocr_conf.mean():.3f}")
    for i, row in out[pred != out.label].iterrows():
        print(f"  WRONG {row.file} (true {'scam' if row.label else 'legit'}, score {row.score:.2f})")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "pipeline_eval.json").write_text(out.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
