"""Step 3: load, clean, merge and split the text datasets.

Run from the project root:   python nlp/prepare_data.py

Inputs  (data/raw/):
  SMSSpamCollection   tab-separated: label<TAB>text   (ham / spam)
  my_scams.csv        columns: text,label             (1 = scam, 0 = legit)

Outputs (data/processed/):
  all_data.csv, train.csv, test.csv   columns: text, clean_text, label, source
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean import clean_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def load_sms():
    path = RAW / "SMSSpamCollection.csv"
    if not path.exists():
        print(f"[skip] {path} not found")
        return None
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                     encoding="utf-8", quoting=3)  # quoting=3: don't treat " as special
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["source"] = "sms_spam_collection"
    return df


def load_mine():
    path = RAW / "my_scams.csv"
    if not path.exists():
        print(f"[skip] {path} not found")
        return None
    df = pd.read_csv(path)
    df["source"] = "my_scams"
    return df


def main():
    frames = [f for f in (load_sms(), load_mine()) if f is not None]
    if not frames:
        sys.exit("No data found in data/raw/. Upload the dataset files first.")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset="clean_text").reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate messages")

    print("\nRows per source and label (0 = legit, 1 = scam):")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0))
    print(f"\nTotal: {len(df)} | scam share: {df['label'].mean():.1%}")

    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "all_data.csv", index=False)
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    print(f"\nSaved train ({len(train)}) and test ({len(test)}) to {OUT}")


if __name__ == "__main__":
    main()
