import glob
import pandas as pd

# 1. Load
path = "data/urldata.csv"
df = pd.read_csv(path)
print("Original:", df.shape)

# 2. Drop useless index column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# 3. Missing values and extra spaces
df = df.dropna(subset=["url", "result"])
df["url"] = df["url"].astype(str).str.strip()
df = df[df["url"] != ""]

# 4. Duplicates
df = df.drop_duplicates(subset="url")

# 5. Label ko 0/1 integer banao
df["result"] = df["result"].astype(int)

print("After cleaning:", df.shape)
print(df["result"].value_counts())

# 6. Save
df[["url", "result"]].to_csv("data/clean_urls.csv", index=False)
print("Saved: data/clean_urls.csv")