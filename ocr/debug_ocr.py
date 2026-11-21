"""Show what the OCR engine actually reads for screenshots that contain links.

    python ocr/debug_ocr.py --engine easyocr

Use this when "URL recall" in evaluate_ocr.py is low: it prints the true link, the link
the OCR found, and the full OCR text, so you can see which characters go wrong.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ocr"))
sys.path.insert(0, str(ROOT / "nlp"))
import extract as ex  # noqa: E402
from clean import extract_urls  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="easyocr", choices=["easyocr", "tesseract"])
ap.add_argument("--n", type=int, default=6, help="how many screenshots to show")
a = ap.parse_args()

shots = ROOT / "data" / "screenshots"
df = pd.read_csv(shots / "labels.csv")
shown = 0
for _, r in df.iterrows():
    urls = extract_urls(r.text)
    if not urls:
        continue
    out = ex.extract_text(shots / r.filename, mode="none", engine=a.engine)
    print(f"\n{r.filename}\n  true link : {urls}\n  OCR links : {out['urls']}\n  OCR text  : {out['text']}")
    if shown == 0 and a.engine == "easyocr":
        boxes = ex._reader.readtext(ex.load_image(shots / r.filename))
        print("  raw boxes : ", [(t, round(float(c), 2)) for _, t, c in boxes])
    shown += 1
    if shown >= a.n:
        break
