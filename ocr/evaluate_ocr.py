"""Measure how well OCR reads the screenshots in data/screenshots (needs labels.csv).

    python ocr/evaluate_ocr.py                 # auto engine
    python ocr/evaluate_ocr.py --engine tesseract

Tries each preprocessing mode so you can see which one works best, and saves
reports/ocr_eval.json. Metrics compare OCR output with the true message text:
  char recall  - share of the true characters found in the OCR text (in order)
  word recall  - share of the true words found in the OCR text
  URL recall   - share of true links read exactly right (important: links go to the URL model)
The OCR text also contains the sender name and time from the screenshot, so we measure
recall (was the message found?) rather than exact match.
"""
import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ocr"))
sys.path.insert(0, str(ROOT / "nlp"))
from clean import extract_urls  # noqa: E402
from extract import extract_text  # noqa: E402

SHOTS = ROOT / "data" / "screenshots"


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def char_recall(truth, out):
    t, o = norm(truth), norm(out)
    sm = difflib.SequenceMatcher(None, t, o, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / max(len(t), 1)


def word_recall(truth, out):
    pool = norm(out).split()
    words = norm(truth).split()
    hit = 0
    for w in words:
        if w in pool:
            pool.remove(w)
            hit += 1
    return hit / max(len(words), 1)


def url_recall(truth, out):
    urls = [u.lower().rstrip("/") for u in extract_urls(truth)]
    if not urls:
        return None
    found = [u.lower().rstrip("/") for u in extract_urls(out)]
    flat = norm(out).replace(" ", "")
    return sum(u in found or u in flat for u in urls) / len(urls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="auto", choices=["auto", "easyocr", "tesseract"])
    args = ap.parse_args()

    labels = SHOTS / "labels.csv"
    if not labels.exists():
        sys.exit("data/screenshots/labels.csv not found. Run ocr/make_synthetic_screenshots.py first.")
    df = pd.read_csv(labels)
    results = {}
    for mode in ("none", "gray", "binary"):
        start = time.time()
        rows = []
        for _, r in df.iterrows():
            out = extract_text(SHOTS / r["filename"], mode=mode, engine=args.engine)
            rows.append((r["filename"], char_recall(r["text"], out["text"]),
                         word_recall(r["text"], out["text"]),
                         url_recall(r["text"], out["text"]), out["confidence"], out["engine"]))
        cr = np.array([x[1] for x in rows])
        wr = np.array([x[2] for x in rows])
        ur = [x[3] for x in rows if x[3] is not None]
        results[mode] = {"char_recall": float(cr.mean()), "word_recall": float(wr.mean()),
                         "url_recall": float(np.mean(ur)), "mean_confidence": float(np.mean([x[4] for x in rows])),
                         "images_below_90pct": int((cr < 0.9).sum()), "seconds": round(time.time() - start, 1),
                         "engine": rows[0][5]}
        worst = sorted(rows, key=lambda x: x[1])[:3]
        results[mode]["worst"] = [(w[0], round(w[1], 3)) for w in worst]

    engine = next(iter(results.values()))["engine"]
    print(f"Engine: {engine} | {len(df)} images\n")
    print(f"{'mode':8s}{'char':>8s}{'word':>8s}{'URL':>8s}{'conf':>8s}{'<90%':>6s}{'sec':>7s}")
    for mode, m in results.items():
        print(f"{mode:8s}{m['char_recall']:8.3f}{m['word_recall']:8.3f}{m['url_recall']:8.3f}"
              f"{m['mean_confidence']:8.3f}{m['images_below_90pct']:6d}{m['seconds']:7.1f}")
    best = max(results, key=lambda k: (results[k]["char_recall"], results[k]["url_recall"]))
    print(f"\nBest mode: {best}. Worst images there: {results[best]['worst']}")
    REPORTS = ROOT / "reports"
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "ocr_eval.json").write_text(json.dumps(results, indent=2))
    print("Saved reports/ocr_eval.json")


if __name__ == "__main__":
    main()
