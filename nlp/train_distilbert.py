"""Step 5: fine-tune DistilBERT as the second text model.

Run in Google Colab with a GPU (Runtime > Change runtime type > T4 GPU):
    python nlp/train_distilbert.py

Needs   data/processed/train.csv and test.csv  (run nlp/prepare_data.py first)
Saves   models/distilbert_scam/  and  reports/distilbert_metrics.json
Compares against reports/tfidf_metrics.json if it exists.
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "distilbert_scam"
REPORTS = ROOT / "reports"

MODEL_NAME = "distilbert-base-uncased"
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_LEN = 128
INDIA_WEIGHT = 3  # same idea as in train_tfidf.py
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(texts, labels, weights, tokenizer, shuffle):
    items = list(zip(texts, labels, weights))

    def collate(batch):
        t, l, w = zip(*batch)
        enc = tokenizer(list(t), truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt")
        return (enc["input_ids"], enc["attention_mask"],
                torch.tensor(l, dtype=torch.long),
                torch.tensor(w, dtype=torch.float))

    return DataLoader(items, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict_proba(model, tokenizer, texts, device):
    """Return P(scam) for each text."""
    model.eval()
    probs = []
    for i in range(0, len(texts), 64):
        enc = tokenizer(list(texts[i:i + 64]), truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(device)
        logits = model(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"]).logits
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    return np.array(probs)


def main():
    for name in ("train.csv", "test.csv"):
        if not (PROC / name).exists():
            sys.exit(f"{PROC / name} not found. Run nlp/prepare_data.py first.")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU found. Training will be very slow. "
              "In Colab: Runtime > Change runtime type > T4 GPU, then rerun.")

    train = pd.read_csv(PROC / "train.csv")
    test = pd.read_csv(PROC / "test.csv")
    print(f"Train: {len(train)} | Test: {len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # BERT does its own lowercasing, so feed it the original text (keeps punctuation)
    y = train["label"].to_numpy()
    sample_w = np.where(train["source"] == "my_scams", INDIA_WEIGHT, 1.0)
    loader = make_loader(train["text"].tolist(), y.tolist(), sample_w.tolist(),
                         tokenizer, shuffle=True)

    # class weights, because only ~13% of messages are scams
    class_w = torch.tensor(len(y) / (2 * np.bincount(y)), dtype=torch.float).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for input_ids, mask, labels, w in loader:
            input_ids, mask = input_ids.to(device), mask.to(device)
            labels, w = labels.to(device), w.to(device)
            logits = model(input_ids=input_ids, attention_mask=mask).logits
            loss = (F.cross_entropy(logits, labels, reduction="none")
                    * class_w[labels] * w).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        print(f"Epoch {epoch}/{EPOCHS} | avg loss {running / len(loader):.4f} "
              f"| {time.time() - start:.0f}s elapsed")

    # Evaluate once, on the held-out test set (no tuning on it)
    probs = predict_proba(model, tokenizer, test["text"].tolist(), device)
    preds = (probs >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        test["label"], preds, average="binary", zero_division=0)
    cm = confusion_matrix(test["label"], preds)
    print("\n=== DistilBERT test results (scam = positive class) ===")
    print(classification_report(test["label"], preds,
                                target_names=["legit", "scam"], digits=3))
    print("Confusion matrix (rows = actual, cols = predicted):")
    print("            pred legit  pred scam")
    print(f"actual legit   {cm[0][0]:6d}     {cm[0][1]:6d}")
    print(f"actual scam    {cm[1][0]:6d}     {cm[1][1]:6d}")

    samples = [
        "Your KYC has expired. Update now at http://sbi-kyc.xyz or account will be blocked",
        "Congratulations you won Rs 25 lakh in KBC lottery. Call now to claim",
        "Earn Rs 5000 daily working from home. Join now",
        "Rs 500 debited from your account at Amazon on 12 Nov",
        "Your OTP for login is 482913. Do not share it with anyone",
        "Hey are we still meeting at 5 today?",
    ]
    print("\nSanity check on Indian-style messages:")
    for s, pr in zip(samples, predict_proba(model, tokenizer, samples, device)):
        print(f"  {pr:.2f}  {s[:70]}")

    REPORTS.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    (REPORTS / "distilbert_metrics.json").write_text(json.dumps({
        "model": "distilbert-base-uncased", "precision": p, "recall": r, "f1": f1,
        "confusion_matrix": cm.tolist(), "n_test": len(test)}, indent=2))
    print(f"\nSaved model to {MODEL_DIR}")

    tfidf_file = REPORTS / "tfidf_metrics.json"
    if tfidf_file.exists():
        t = json.loads(tfidf_file.read_text())
        print("\n=== Comparison (scam class, same test set) ===")
        print(f"{'':14s}{'precision':>10s}{'recall':>9s}{'F1':>8s}")
        print(f"{'TF-IDF + LR':14s}{t['precision']:10.3f}{t['recall']:9.3f}{t['f1']:8.3f}")
        print(f"{'DistilBERT':14s}{p:10.3f}{r:9.3f}{f1:8.3f}")


if __name__ == "__main__":
    main()
