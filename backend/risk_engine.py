WEIGHTS = {"text": 0.6, "url": 0.4}


def compute_risk(text_score=None, url_score=None):
    scores = {"text": text_score, "url": url_score}
    available = {k: v for k, v in scores.items() if v is not None}

    # Kuch bhi check nahi hua: "legit" nahi, unknown
    if not available:
        return {"risk_score": None, "verdict": None}

    total_weight = sum(WEIGHTS[k] for k in available)
    risk = sum(WEIGHTS[k] * v for k, v in available.items()) / total_weight
    risk = round(risk, 3)

    if risk >= 0.7:
        verdict = "scam"
    elif risk >= 0.4:
        verdict = "suspicious"
    else:
        verdict = "legit"

    return {"risk_score": risk, "verdict": verdict}