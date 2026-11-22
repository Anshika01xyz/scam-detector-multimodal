import os
import sys
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Let Python find your url_model folder and risk_engine.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "url_model"))

from risk_engine import compute_risk
import features  # your url_model/features.py

app = FastAPI(title="Scam Detector API")

bundle = joblib.load(os.path.join(BASE_DIR, "url_model", "url_model.joblib"))
model = bundle["model"]


class UrlRequest(BaseModel):
    url: str


class AnalyzeRequest(BaseModel):
    text_score: Optional[float] = None
    text_verdict: Optional[str] = None
    scam_type: Optional[str] = None
    reasons: Optional[list] = None
    extracted_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    urls_found: Optional[list] = None
    url: Optional[str] = None


def predict_url_score(url: str) -> float:
    # Well-known official domains get a low score (rule layer on top of the model)
    if features.is_trusted(url):
        return 0.01

    feats = features.extract_features(url)
    X = pd.DataFrame([feats])
    cols = bundle.get("columns")
    if cols:
        X = X[list(cols)]
    elif hasattr(model, "feature_names_in_"):
        X = X[list(model.feature_names_in_)]
    return float(model.predict_proba(X)[0][1])


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/predict-url")
def predict_url(req: UrlRequest):
    score = predict_url_score(req.url)
    label = "phishing" if score >= 0.5 else "legit"
    return {
        "label": label,
        "score": round(score, 3),
        "reasons": features.get_reasons(req.url),
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    urls = list(req.urls_found or [])
    if req.url:
        urls.append(req.url)

    url_results = []
    for u in urls:
        s = predict_url_score(u)
        url_results.append({
            "url": u,
            "label": "phishing" if s >= 0.5 else "legit",
            "score": round(s, 3),
            "reasons": features.get_reasons(u),
        })

    # The riskiest URL is used in the final risk score
    url_score = max((r["score"] for r in url_results), default=None)

    result = compute_risk(req.text_score, url_score)

    reasons = list(req.reasons or [])
    for r in url_results:
        if r["label"] == "phishing":
            reasons.extend(r["reasons"])

    is_legit = result["verdict"] == "legit"
    result["scam_type"] = None if is_legit else req.scam_type
    result["reasons"] = [] if is_legit else reasons
    result["url_results"] = url_results
    return result