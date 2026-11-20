def compute_risk(text_score=None, screenshot_score=None, url_score=None):
    weights = {"text": 0.4, "screenshot": 0.3, "url": 0.3}
    scores = {"text": text_score, "screenshot": screenshot_score, "url": url_score}

    available = {k: v for k, v in scores.items() if v is not None}
    if not available:
        return {"risk_score": 0, "verdict": "safe"}

    total_weight = sum(weights[k] for k in available)
    risk = sum(weights[k] * v for k, v in available.items()) / total_weight
    risk_score = round(risk * 100)

    if risk_score >= 70:
        verdict = "scam"
    elif risk_score >= 40:
        verdict = "suspicious"
    else:
        verdict = "safe"

    return {"risk_score": risk_score, "verdict": verdict}