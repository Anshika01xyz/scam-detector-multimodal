"""Train and compare the course (UML501) classifiers on the scam-text task.

    python nlp/prepare_data.py        # once, creates data/processed/train.csv + test.csv
    python nlp/train_classical.py

Models (all from the syllabus): Logistic Regression, Naive Bayes, K-NN, SVM, Random Forest,
AdaBoost, XGBoost (skipped if xgboost is not installed) and a shallow neural network (MLP).

Same features for every model: TF-IDF (1-2 grams) fitted on the training set only.
Same data: Indian-style rows (source == "my_scams") are repeated 3x in training, like INDIA_WEIGHT
in train_tfidf.py, so that every model sees the same training set (K-NN and MLP cannot take
sample weights). Class imbalance (~15% scams) is handled with class_weight / scale_pos_weight where
the model supports it.

Evaluated on data/processed/test.csv (SMS + Indian-style) and data/test/hard_test.csv (160 messages,
never trained on). Metrics: accuracy, precision, recall (= sensitivity), specificity, F1.
Saves reports/model_comparison.csv and reports/model_comparison.json.
"""
import json
import time
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean import clean_text  # noqa: E402

try:
    from xgboost import XGBClassifier
except ImportError:  # keep the script usable without xgboost
    XGBClassifier = None

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
HARD = ROOT / "data" / "test" / "hard_test.csv"
REPORTS = ROOT / "reports"
INDIA_REPEAT = 3
SEED = 42


def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "specificity": spec, "f1": f1, "fp": int(fp), "fn": int(fn)}


def make_models(n_neg, n_pos):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "K-NN (k=5, cosine)": KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance"),
        "SVM (linear)": SVC(kernel="linear", class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                n_jobs=-1, random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=SEED),
        "Neural net (MLP)": MLPClassifier(hidden_layer_sizes=(128,), max_iter=200,
                                          early_stopping=True, random_state=SEED),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                          scale_pos_weight=n_neg / n_pos, eval_metric="logloss",
                                          n_jobs=-1, random_state=SEED)
    else:
        print("[skip] xgboost not installed (pip install xgboost) - XGBoost left out\n")
    return models


def main():
    train = pd.read_csv(PROC / "train.csv")
    test = pd.read_csv(PROC / "test.csv")
    hard = pd.read_csv(HARD)
    hard["clean_text"] = hard["text"].apply(clean_text)

    india = train[train["source"] == "my_scams"]
    train_aug = pd.concat([train] + [india] * (INDIA_REPEAT - 1), ignore_index=True)
    print(f"Train: {len(train)} rows ({len(train_aug)} after repeating Indian rows {INDIA_REPEAT}x) | "
          f"Test: {len(test)} | Hard test: {len(hard)}\n")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_tr = vec.fit_transform(train_aug["clean_text"])
    X_te, X_hd = vec.transform(test["clean_text"]), vec.transform(hard["clean_text"])
    y_tr = train_aug["label"].to_numpy()
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos

    rows = []
    for name, model in make_models(n_neg, n_pos).items():
        t0 = time.time()
        Xfit = X_tr.toarray() if name == "AdaBoost" else X_tr   # AdaBoost stumps need dense input
        model.fit(Xfit, y_tr)
        fit_s = time.time() - t0
        conv = (lambda X: X.toarray()) if name == "AdaBoost" else (lambda X: X)
        for split, X, y in (("test", X_te, test["label"]), ("hard", X_hd, hard["label"])):
            m = metrics(y.to_numpy(), model.predict(conv(X)))
            rows.append({"model": name, "split": split, "train_seconds": round(fit_s, 1), **m})
        print(f"done: {name} ({fit_s:.1f}s)")

    df = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "model_comparison.csv", index=False)
    (REPORTS / "model_comparison.json").write_text(json.dumps(rows, indent=2))
    for split, title in (("test", "MAIN TEST SET"), ("hard", "HARD TEST SET")):
        d = df[df["split"] == split].drop(columns="split").sort_values("f1", ascending=False)
        print(f"\n=== {title} (scam = positive class) ===")
        print(d.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved {REPORTS / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
