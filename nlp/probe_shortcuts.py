"""Does the model treat "contains a link" as proof of a scam?

For each message we score it twice: without a link and with a link added. A model that
reads the wording should barely change; a model using the shortcut jumps up.

    python nlp/probe_shortcuts.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_indian import BERT_DIR, TFIDF, score_distilbert, score_tfidf  # noqa: E402

LEGIT = [
    ("Call starts at 4 PM today.", "https://meet.google.com/abc-defg-hij"),
    ("Your parcel is out for delivery today.", "https://www.delhivery.com/track/12345"),
    ("Sale ends tonight, get 40% off on all shoes.", "https://www.ajio.com/sale"),
    ("Your bank statement for October is ready.", "https://www.hdfcbank.com/statement"),
    ("Class notes for the exam are shared with you.", "https://drive.google.com/file/d/1abc/view"),
    ("Your electricity bill for November is generated.", "https://www.tatapower.com/bill"),
    ("Your ticket is confirmed. Journey date 25 Dec.", "https://www.irctc.co.in/tickets"),
    ("The team lunch menu is here, please vote.", "https://forms.gle/xYz123"),
]
SCAM = [
    ("Your account will be blocked today. Verify your PAN now.", "http://bank-verify.top"),
    ("You have won a prize. Pay the fee to claim it.", "http://prize-claim.xyz"),
    ("Pay a small fee to release your parcel from customs.", "http://parcel-fee.click"),
    ("Your KYC has expired. Update it to keep your wallet active.", "https://wallet-kyc.in.net"),
    ("Earn Rs 4000 daily from home. Join our group.", "t.me/earn_daily_india"),
    ("Your refund is ready. Enter your card details to receive it.", "http://refund-now.shop"),
    ("Your SIM will be disconnected tonight. Complete verification.", "http://sim-verify.online"),
    ("Final notice: electricity will be cut today. Pay the pending bill.", "http://power-pay.xyz"),
]


def main():
    models = {}
    if TFIDF.exists():
        models["TF-IDF"] = score_tfidf
    if BERT_DIR.exists():
        models["DistilBERT"] = score_distilbert
    if not models:
        sys.exit("No trained model found. Run nlp/train_tfidf.py (and train_distilbert.py) first.")
    for label, pairs in (("LEGIT messages", LEGIT), ("SCAM messages", SCAM)):
        plain = [p[0] for p in pairs]
        linked = [f"{p[0]} {p[1]}" for p in pairs]
        print(f"\n=== {label}: score without link -> with link ===")
        for name, fn in models.items():
            a, b = np.asarray(fn(plain)), np.asarray(fn(linked))
            print(f"{name:11s} average {a.mean():.2f} -> {b.mean():.2f}   (change {b.mean() - a.mean():+.2f})")
            for t, x, y in zip(plain, a, b):
                print(f"     {x:.2f} -> {y:.2f}  {t[:60]}")
    print("\nGood sign: legit messages stay low with a link (change near 0). "
          "A big jump means the model treats links as a scam signal.")


if __name__ == "__main__":
    main()
