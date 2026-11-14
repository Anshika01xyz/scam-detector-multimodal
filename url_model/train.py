import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix)
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

# 3. Split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, eval_metric="logloss", n_jobs=-1),
}

best_name, best_f1, best_model = None, -1, None
for name, m in models.items():
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    f1 = f1_score(y_te, pred)
    print(f"\n{name}")
    print("  Precision:", round(precision_score(y_te, pred), 4))
    print("  Recall   :", round(recall_score(y_te, pred), 4))
    print("  F1       :", round(f1, 4))
    print("  False positive rate:", round(fp / (fp + tn), 4))
    print("  Confusion matrix [[TN, FP], [FN, TP]]:", [[tn, fp], [fn, tp]])
    if f1 > best_f1:
        best_name, best_f1, best_model = name, f1, m

# 5. Best model save
joblib.dump({"model": best_model, "columns": list(X.columns)}, "url_model/url_model.joblib")
print("\nBest model:", best_name, "-> saved url_model/url_model.joblib")