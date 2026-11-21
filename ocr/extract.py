"""Extract text from a screenshot.

    from ocr.extract import extract_text
    result = extract_text("data/screenshots/shot_01_scam.jpg")
    # {"text": "...", "confidence": 0.93, "urls": [...], "engine": "easyocr", "mode": "binary"}

Engines: "easyocr" (main, deep-learning) or "tesseract" (fallback, needs the tesseract
program installed). "auto" uses EasyOCR if installed, otherwise tesseract.
Preprocess modes: "auto" (default), "none" (original image), "gray", "binary" (see ocr/preprocess.py).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ocr"))
sys.path.insert(0, str(ROOT / "nlp"))
from clean import extract_urls  # noqa: E402
from preprocess import load_image, preprocess  # noqa: E402

DEFAULT_MODE = "auto"  # auto: "none" for EasyOCR, "binary" for tesseract (best on our tests)
_reader = None


def _pick_engine(engine):
    if engine != "auto":
        return engine
    try:
        import easyocr  # noqa: F401
        return "easyocr"
    except ImportError:
        return "tesseract"


def _easyocr_read(img):
    global _reader
    if _reader is None:
        import easyocr
        try:
            import torch
            gpu = torch.cuda.is_available()
        except ImportError:
            gpu = False
        _reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    items = []
    for box, text, conf in _reader.readtext(img, detail=1, paragraph=False):
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        if text.strip():
            items.append((min(ys), max(ys) - min(ys), min(xs), text.strip(), float(conf)))
    if not items:
        return "", 0.0
    # put boxes into reading order: group into lines by vertical position, then left to right
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
    text = " ".join(t[3] for t in ordered)
    weights = [len(t[3]) for t in ordered]
    conf = float(np.average([t[4] for t in ordered], weights=weights))
    return text, conf


def _tesseract_read(img):
    import pytesseract
    data = pytesseract.image_to_data(img, config="--psm 6", output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for w, c in zip(data["text"], data["conf"]):
        if w.strip() and float(c) >= 0:
            words.append(w.strip())
            confs.append(float(c) / 100)
    return " ".join(words), (float(np.mean(confs)) if confs else 0.0)


def extract_text(image, mode=DEFAULT_MODE, engine="auto"):
    """image: file path or a BGR numpy array. Returns dict with text, confidence, urls."""
    img = load_image(image) if isinstance(image, (str, Path)) else image
    engine = _pick_engine(engine)
    if mode == "auto":
        mode = "none" if engine == "easyocr" else "binary"
    if mode == "gray":
        img = preprocess(img)
    elif mode == "binary":
        img = preprocess(img, binarize=True)
    elif mode != "none":
        raise ValueError("mode must be 'auto', 'none', 'gray' or 'binary'")
    text, conf = _easyocr_read(img) if engine == "easyocr" else _tesseract_read(img)
    return {"text": text, "confidence": round(conf, 3), "urls": extract_urls(text),
            "engine": engine, "mode": mode}
