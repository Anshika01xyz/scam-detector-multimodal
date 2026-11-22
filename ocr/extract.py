"""Extract text from a screenshot.

    from extract import extract_text
    result = extract_text("data/screenshots/shot_01_scam.jpg")
    # {"text": "...", "confidence": 0.93, "urls": [...], "engine": "tesseract", "mode": "dual"}

Engines
  "tesseract"  needs the tesseract program (apt install tesseract-ocr) + pip install pytesseract
  "easyocr"    deep-learning engine, pip install easyocr
  "auto"       tesseract if available, otherwise EasyOCR

Preprocess modes (see ocr/preprocess.py)
  "none" original image | "gray" | "binary" black-and-white |
  "dual" (tesseract only): reads the text from the binary image and the links from the
         original image, because on our tests binary reads text best but links worse.
  "auto" -> "dual" for tesseract, "none" for EasyOCR.

strip_ui=True removes the phone's status bar / chat header (top 12% of tall screenshots) and the
time stamp at the end, so only the message itself reaches the classifier.
"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ocr"))
sys.path.insert(0, str(ROOT / "nlp"))
from clean import extract_urls  # noqa: E402
from preprocess import load_image, preprocess  # noqa: E402

DEFAULT_MODE = "auto"
CHROME_TOP = 0.12  # fraction of a tall screenshot that is status bar + chat header
TRAILING_TIME = re.compile(r"\s*\b\d{1,2}(?:[:.]\s?\d{2}\s?(?:[ap][mi]?)?|\s\d{2}\s?[ap][mi])\s*$", re.I)
_reader = None


def _tesseract_ok():
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
        return True
    except ImportError:
        return False


def _easyocr_ok():
    try:
        import easyocr  # noqa: F401
        return True
    except ImportError:
        return False


def _pick_engine(engine):
    if engine != "auto":
        return engine
    if _tesseract_ok():
        return "tesseract"
    if _easyocr_ok():
        return "easyocr"
    raise RuntimeError("No OCR engine found. Install tesseract (+pytesseract) or easyocr.")


def _is_tall(img):
    h, w = img.shape[:2]
    return h / max(w, 1) >= 1.3


def _finish(text, conf, strip_ui):
    text = re.sub(r"\s+", " ", text).strip()
    if strip_ui:
        text = TRAILING_TIME.sub("", text).strip()
    return text, conf


def _easyocr_read(img, strip_ui):
    global _reader
    if _reader is None:
        import easyocr
        try:
            import torch
            gpu = torch.cuda.is_available()
        except ImportError:
            gpu = False
        _reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    H = img.shape[0]
    cut = CHROME_TOP * H if (strip_ui and _is_tall(img)) else -1
    items = []
    for box, text, conf in _reader.readtext(img, detail=1, paragraph=False):
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        if text.strip() and min(ys) >= cut:
            items.append((min(ys), max(ys) - min(ys), min(xs), text.strip(), float(conf)))
    if not items:
        return "", 0.0
    # reading order: group boxes into lines by vertical position, then left to right
    items.sort(key=lambda t: t[0])
    tol = 0.6 * float(np.median([t[1] for t in items]))
    lines, cur = [], [items[0]]
    for it in items[1:]:
        if abs(it[0] - cur[0][0]) <= tol:
            cur.append(it)
        else:
            lines.append(cur)
            cur = [it]
    lines.append(cur)
    ordered = [it for line in lines for it in sorted(line, key=lambda t: t[2])]
    conf = float(np.average([t[4] for t in ordered], weights=[len(t[3]) for t in ordered]))
    return _finish(" ".join(t[3] for t in ordered), conf, strip_ui)


def _tesseract_read(img, strip_ui):
    import pytesseract
    data = pytesseract.image_to_data(img, config="--psm 6", output_type=pytesseract.Output.DICT)
    cut = CHROME_TOP * img.shape[0] if (strip_ui and _is_tall(img)) else -1
    words, confs = [], []
    for w, c, top in zip(data["text"], data["conf"], data["top"]):
        if w.strip() and float(c) >= 0 and top >= cut:
            words.append(w.strip())
            confs.append(float(c) / 100)
    return _finish(" ".join(words), (float(np.mean(confs)) if confs else 0.0), strip_ui)


def _read(engine, img, strip_ui):
    return (_easyocr_read if engine == "easyocr" else _tesseract_read)(img, strip_ui)


def _squash(s):
    return re.sub(r"\s+", "", s.lower())


def extract_text(image, mode=DEFAULT_MODE, engine="auto", strip_ui=True):
    """image: file path or a BGR numpy array. Returns dict with text, confidence, urls."""
    img = load_image(image) if isinstance(image, (str, Path)) else image
    engine = _pick_engine(engine)
    if mode == "auto":
        mode = "dual" if engine == "tesseract" else "none"
    if mode == "dual" and engine != "tesseract":
        mode = "none"

    if mode == "dual":
        text, conf = _read(engine, preprocess(img, binarize=True), strip_ui)
        text_raw, _ = _read(engine, img, strip_ui)
        urls = extract_urls(text_raw) or extract_urls(text)
        for u in urls:  # make sure the correctly-read link is in the text the model sees
            if _squash(u) not in _squash(text):
                text = f"{text} {u}"
    else:
        if mode == "gray":
            work = preprocess(img)
        elif mode == "binary":
            work = preprocess(img, binarize=True)
        elif mode == "none":
            work = img
        else:
            raise ValueError("mode must be 'auto', 'none', 'gray', 'binary' or 'dual'")
        text, conf = _read(engine, work, strip_ui)
        urls = extract_urls(text)
    return {"text": text, "confidence": round(conf, 3), "urls": urls, "engine": engine, "mode": mode}
