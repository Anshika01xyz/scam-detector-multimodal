"""One function that turns text or a screenshot into the JSON of docs/interface.md (message 1).

    from predict import analyze
    analyze(text="Your KYC has expired. Update now at http://sbi-kyc.xyz")
    analyze(image="data/screenshots/shot_01_scam.jpg")

Command line:
    python nlp/predict.py "Your KYC has expired..."
    python nlp/predict.py --image data/screenshots/shot_01_scam.jpg --model tfidf

Model: "auto" uses DistilBERT if models/distilbert_scam exists, otherwise TF-IDF; "ensemble" averages both.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nlp"))
sys.path.insert(0, str(ROOT / "ocr"))
from clean import clean_text, extract_urls  # noqa: E402
from reasons import get_reasons  # noqa: E402
from scam_type import detect_scam_type  # noqa: E402

TFIDF = ROOT / "models" / "tfidf_lr.pkl"
BERT_DIR = ROOT / "models" / "distilbert_scam"
_cache = {}


def verdict_from_score(score):
    """legit < 0.40, suspicious 0.40 to 0.70, scam > 0.70 (docs/interface.md)."""
    if score < 0.40:
        return "legit"
    return "suspicious" if score <= 0.70 else "scam"


def resolve_model(model):
    """"auto": DistilBERT if trained, else TF-IDF. "ensemble": average of both (if both exist)."""
    have_bert, have_tfidf = BERT_DIR.exists(), TFIDF.exists()
    if model == "auto":
        return "distilbert" if have_bert else "tfidf"
    if model == "ensemble":
        return "ensemble" if (have_bert and have_tfidf) else ("distilbert" if have_bert else "tfidf")
    return model


def score_text(text, model="auto"):
    """Probability (0 to 1) that the text is a scam."""
    model = resolve_model(model)
    if model == "ensemble":
        return (score_text(text, "tfidf") + score_text(text, "distilbert")) / 2
    if model == "tfidf":
        if "tfidf" not in _cache:
            _cache["tfidf"] = joblib.load(TFIDF)
        return float(_cache["tfidf"].predict_proba([clean_text(text)])[0, 1])
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if "bert" not in _cache:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(BERT_DIR)
        mdl = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(device).eval()
        _cache["bert"] = (tok, mdl, device)
    tok, mdl, device = _cache["bert"]
    with torch.no_grad():
        enc = tok([text], truncation=True, max_length=128, return_tensors="pt").to(device)
        logits = mdl(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
    return float(torch.softmax(logits, dim=-1)[0, 1])


def analyze(text=None, image=None, model="auto", ocr_engine="auto", ocr_mode="auto"):
    """Return the Person 1 -> Person 2 message (all keys always present)."""
    if (text is None) == (image is None):
        raise ValueError("Give exactly one of text= or image=")
    if image is not None:
        from extract import extract_text
        ocr = extract_text(image, mode=ocr_mode, engine=ocr_engine)
        text, conf, input_type = ocr["text"], ocr["confidence"], "screenshot"
    else:
        conf, input_type = None, "text"

    text = (text or "").strip()
    out = {"input_type": input_type, "extracted_text": text, "text_score": None,
           "text_verdict": None, "scam_type": None, "reasons": [],
           "urls_found": extract_urls(text), "ocr_confidence": conf}
    if not text:
        out["reasons"] = ["no readable text found"]
        return out

    score = round(score_text(text, model), 4)
    verdict = verdict_from_score(score)
    out["text_score"], out["text_verdict"] = score, verdict
    if verdict != "legit":
        out["scam_type"] = detect_scam_type(text)
        out["reasons"] = get_reasons(text, out["urls_found"]) or ["wording similar to known scams (model)"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="message text")
    ap.add_argument("--image", help="path to a screenshot instead of text")
    ap.add_argument("--model", default="auto", choices=["auto", "tfidf", "distilbert", "ensemble"])
    a = ap.parse_args()
    result = analyze(text=a.text if not a.image else None, image=a.image, model=a.model)
    print(json.dumps(result, indent=2, ensure_ascii=False))
