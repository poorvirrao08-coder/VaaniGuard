
import re
from pathlib import Path

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load dataset
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "vaaniguard_cleaned_sentinel_dataset.csv")
df = df[["text_cleaned", "binary_label"]].dropna().copy()

df["text_cleaned"] = df["text_cleaned"].astype(str).str.strip()
df = df[df["text_cleaned"] != ""]
df["binary_label"] = df["binary_label"].astype(int)

conflicts = df.groupby("text_cleaned")["binary_label"].nunique()
ambiguous = conflicts[conflicts > 1].index

df = df[~df["text_cleaned"].isin(ambiguous)]
df = df.drop_duplicates(subset="text_cleaned").reset_index(drop=True)

# Recreate the exact group-aware split
def normalize_template(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\b[\w.-]+\.com\b", " URL ", text)
    text = re.sub(r"\b\d+\b", " NUMBER ", text)
    return re.sub(r"\s+", " ", text).strip()

groups = df["text_cleaned"].apply(normalize_template)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, df["binary_label"], groups=groups)
)

train = df.iloc[train_idx]
test = df.iloc[test_idx]

# Verify group separation
train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

print("\nGROUP OVERLAP AUDIT")
print("Shared normalized groups:", len(train_groups & test_groups))

# Character-level similarity
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=100000
)

train_vectors = vectorizer.fit_transform(train["text_cleaned"])
test_vectors = vectorizer.transform(test["text_cleaned"])

nn = NearestNeighbors(
    n_neighbors=1,
    metric="cosine"
)

nn.fit(train_vectors)

distances, indices = nn.kneighbors(test_vectors)
similarities = 1 - distances.flatten()

print("\nNEAR-DUPLICATE AUDIT")
for threshold in [0.90, 0.95, 0.99]:
    count = (similarities >= threshold).sum()
    print(f"Similarity >= {threshold}: {count} test messages")

print("\nTOP 10 MOST SIMILAR PAIRS")

for i in similarities.argsort()[::-1][:10]:
    train_index = indices[i][0]

    print("\nSimilarity:", round(similarities[i], 4))
    print("Test message:", test.iloc[i]["text_cleaned"][:200])
    print("Train message:", train.iloc[train_index]["text_cleaned"][:200])

print("\nGROUP-AWARE AUDIT COMPLETED")
