"""Build a folder (and zip) that can be uploaded to a Hugging Face Space to host the app online.

    python deploy/make_space_bundle.py          ->  space_bundle/  and  space_bundle.zip

Run it after training (it copies the trained models from models/). The scikit-learn version is
pinned to the one used here, because saved TF-IDF models only load in the same version.
"""
import shutil
import sys
import zipfile
from pathlib import Path

import sklearn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "space_bundle"

README = """---
title: Scam Message Detector
emoji: 🛡️
colorFrom: red
colorTo: yellow
sdk: streamlit
app_file: app.py
pinned: false
---

# Scam Message Detector

Paste a message or upload a screenshot. The app reads the text (OCR for screenshots), scores how likely it
is to be a scam, names the scam type and explains why. Study project, results are estimates, not proof.

Report fraud at cybercrime.gov.in or call 1930.

Credits: training data includes the UCI SMS Spam Collection (Almeida & Hidalgo, CC BY 4.0) plus
team-written examples.
"""

ENTRY = '''from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).parent / "frontend" / "app.py"), run_name="__main__")
'''


def main():
    have = [p for p in ("tfidf_lr.pkl", "distilbert_scam") if (ROOT / "models" / p).exists()]
    if not have:
        sys.exit("No trained models in models/. Run nlp/train_tfidf.py (and train_distilbert.py) first.")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    for d in ("nlp", "ocr", "frontend"):
        shutil.copytree(ROOT / d, OUT / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (OUT / "models").mkdir()
    if "tfidf_lr.pkl" in have:
        shutil.copy2(ROOT / "models" / "tfidf_lr.pkl", OUT / "models" / "tfidf_lr.pkl")
    if "distilbert_scam" in have:
        shutil.copytree(ROOT / "models" / "distilbert_scam", OUT / "models" / "distilbert_scam")
    (OUT / "app.py").write_text(ENTRY, encoding="utf-8")
    (OUT / "README.md").write_text(README, encoding="utf-8")
    (OUT / "packages.txt").write_text("tesseract-ocr\n", encoding="utf-8")
    (OUT / "requirements.txt").write_text(
        "--extra-index-url https://download.pytorch.org/whl/cpu\n"
        "torch\ntransformers\n"
        f"scikit-learn=={sklearn.__version__}\n"
        "joblib\npandas\nnumpy\nopencv-python-headless\npillow\npytesseract\n", encoding="utf-8")
    zpath = ROOT / "space_bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in OUT.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(OUT))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"Bundle ready: {OUT} ({size:.0f} MB, models: {', '.join(have)})")
    print(f"Zip: {zpath}")


if __name__ == "__main__":
    main()
