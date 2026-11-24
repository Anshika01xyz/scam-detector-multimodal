import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import extract_features

# 1. Load (sample lete hain taaki time bache)
df = pd.read_csv("data/clean_urls.csv")
df = df.sample(n=min(100000, len(df)), random_state=42)
print("Using rows:", len(df))

# 2. Features banao
t = time.time()
X = pd.DataFrame([extract_features(u) for u in df["url"]])
y = df["result"].values
print("Features done in", round(time.time() - t), "sec")

# 3. Split (40% testing)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y)

# 4. Models
models = {
    "LogisticRegression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, eval_metric="logloss", n_jobs=-1),
}

results = []
best_name, best_f1, best_model = None, -1, None
for name, m in models.items():
    t0 = time.time()
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    acc = accuracy_score(y_te, pred)
    prec = precision_score(y_te, pred)
    rec = recall_score(y_te, pred)
    f1 = f1_score(y_te, pred)
    fpr = fp / (fp + tn)
    results.append((name, acc, prec, rec, f1, fpr))
    print(f"\n{name}  (trained in {round(time.time() - t0)} sec)")
    print("  Accuracy :", round(acc, 4))
    print("  Precision:", round(prec, 4))
    print("  Recall   :", round(rec, 4))
    print("  F1       :", round(f1, 4))
    print("  False positive rate:", round(fpr, 4))
    print("  Confusion matrix [[TN, FP], [FN, TP]]:", [[tn, fp], [fn, tp]])
    if f1 > best_f1:
        best_name, best_f1, best_model = name, f1, m

# Comparison table
print("\n=== Comparison (test set) ===")
print(f"{'Model':20} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'FPR':>7}")
for name, acc, prec, rec, f1, fpr in results:
    print(f"{name:20} {acc:7.4f} {prec:7.4f} {rec:7.4f} {f1:7.4f} {fpr:7.4f}")

# 5. Best model save
joblib.dump({"model": best_model, "columns": list(X.columns)}, "url_model/url_model.joblib")
print("\nBest model:", best_name, "-> saved url_model/url_model.joblib")