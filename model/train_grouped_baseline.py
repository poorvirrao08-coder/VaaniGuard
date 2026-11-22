
from pathlib import Path

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse.csgraph import connected_components

# 1. Load dataset
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "vaaniguard_cleaned_sentinel_dataset.csv")
df = df[["text_cleaned", "binary_label"]].dropna()

df["text_cleaned"] = df["text_cleaned"].astype(str).str.strip()
df = df[df["text_cleaned"] != ""]
df = df.drop_duplicates(subset="text_cleaned").reset_index(drop=True)

X = df["text_cleaned"]
y = df["binary_label"].astype(int)

# 2. Identify similar message families
print("Finding similar message templates...")

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=50000
)

X_char = char_vectorizer.fit_transform(X)

# Similarity >= 0.90 means cosine distance <= 0.10
neighbors = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

neighbors.fit(X_char)

similarity_graph = neighbors.radius_neighbors_graph(
    X_char,
    radius=0.10,
    mode="connectivity"
)

# 3. Assign group IDs using connected components
number_of_groups, group_ids = connected_components(
    similarity_graph,
    directed=False
)

group_sizes = pd.Series(group_ids).value_counts()

print("\nTotal similarity groups:", number_of_groups)
print("Largest group size:", group_sizes.max())
print("Groups containing multiple messages:",
      (group_sizes > 1).sum())

# 4. Split by groups, preserving class proportions
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=group_ids)
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

train_groups = set(group_ids[train_idx])
test_groups = set(group_ids[test_idx])

assert train_groups.isdisjoint(test_groups)

print("\nGROUPED DATA SPLIT")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training class distribution:")
print(y_train.value_counts())
print("Testing class distribution:")
print(y_test.value_counts())
print("Shared groups:", len(train_groups & test_groups))

# 5. Build the baseline model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# 6. Train
print("\nTraining grouped baseline...")
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)

print("\n========== GROUPED MODEL RESULTS ==========")

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Scam"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Save separately
joblib.dump(model, BASE_DIR / "vaaniguard_grouped_baseline.pkl")

print("\nGrouped model saved successfully!")
