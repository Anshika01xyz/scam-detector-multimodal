"""Step 4: TF-IDF + Logistic Regression baseline.

Run from the project root (after prepare_data.py):
    python nlp/train_tfidf.py

Reads  data/processed/train.csv, test.csv
Saves  models/tfidf_lr.pkl  and  reports/tfidf_metrics.json
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
INDIA_WEIGHT = 3  # how much more the Indian-style rows (my_scams*.csv) count during training


def main():
    for name in ("train.csv", "test.csv"):
        if not (PROC / name).exists():
            sys.exit(f"{PROC / name} not found. Run nlp/prepare_data.py first.")

    train = pd.read_csv(PROC / "train.csv")
    test = pd.read_csv(PROC / "test.csv")
    print(f"Train: {len(train)} | Test: {len(test)}")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        # class_weight="balanced" because only ~12% of messages are scams
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    # The Indian examples are few (~120) next to ~5,000 UK-style SMS, so count each
    # of them 3x. Without this the model flags normal Indian bank alerts as scams.
    weights = np.where(train["source"] == "my_scams", INDIA_WEIGHT, 1.0)
    model.fit(train["clean_text"], train["label"], clf__sample_weight=weights)

    probs = model.predict_proba(test["clean_text"])[:, 1]
    preds = (probs >= 0.5).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        test["label"], preds, average="binary", zero_division=0)
    cm = confusion_matrix(test["label"], preds)
    print("\n=== Test results (scam = positive class) ===")
    print(classification_report(test["label"], preds,
                                target_names=["legit", "scam"], digits=3))
    print("Confusion matrix (rows = actual, cols = predicted):")
    print("            pred legit  pred scam")
    print(f"actual legit   {cm[0][0]:6d}     {cm[0][1]:6d}")
    print(f"actual scam    {cm[1][0]:6d}     {cm[1][1]:6d}")

    # Which words push the model towards "scam"?
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    words = np.array(vec.get_feature_names_out())
    top = np.argsort(clf.coef_[0])[-15:][::-1]
    print("\nTop scam words:", ", ".join(words[top]))

    # Quick check on Indian-style messages the dataset barely covers
    samples = [
        "Your KYC has expired. Update now at http://sbi-kyc.xyz or account will be blocked",
        "Congratulations you won Rs 25 lakh in KBC lottery. Call now to claim",
        "Earn Rs 5000 daily working from home. Join now",
        "Rs 500 debited from your account at Amazon on 12 Nov",
        "Your OTP for login is 482913. Do not share it with anyone",
        "Hey are we still meeting at 5 today?",
    ]
    print("\nSanity check on Indian-style messages:")
    for s, pr in zip(samples, model.predict_proba(samples)[:, 1]):
        print(f"  {pr:.2f}  {s[:70]}")

    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "tfidf_lr.pkl")
    (REPORTS / "tfidf_metrics.json").write_text(json.dumps({
        "model": "tfidf_logreg", "precision": p, "recall": r, "f1": f1,
        "confusion_matrix": cm.tolist(), "n_test": len(test)}, indent=2))
    print(f"\nSaved model to {MODELS / 'tfidf_lr.pkl'}")


if __name__ == "__main__":
    main()
