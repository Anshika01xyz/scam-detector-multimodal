import pandas as pd
import sys
import os
from typing import Optional

import joblib
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
    url: Optional[str] = None
    text_score: Optional[float] = None
    screenshot_score: Optional[float] = None


def predict_url_score(url: str) -> float:
    feats = features.extract_features(url)
    X = pd.DataFrame([feats])
    cols = bundle.get("features") or bundle.get("feature_names")
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
    label = "phishing" if score >= 0.5 else "legitimate"
    return {
        "label": label,
        "score": round(score, 3),
        "reasons": features.get_reasons(req.url),
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    url_score = predict_url_score(req.url) if req.url else None
    return compute_risk(req.text_score, req.screenshot_score, url_score)