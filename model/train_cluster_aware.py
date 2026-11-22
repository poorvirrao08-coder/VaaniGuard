
from pathlib import Path

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import radius_neighbors_graph
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from scipy.sparse.csgraph import connected_components

# 1. Load dataset
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "vaaniguard_cleaned_sentinel_dataset.csv")
df = df[["text_cleaned", "binary_label"]].dropna().copy()

df["text_cleaned"] = df["text_cleaned"].astype(str).str.strip()
df = df[df["text_cleaned"] != ""]
df = df.drop_duplicates(subset="text_cleaned").reset_index(drop=True)

print("Total messages:", len(df))

# 2. Create similarity groups
print("\nBuilding similarity groups...")

similarity_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=100000
)

similarity_vectors = similarity_vectorizer.fit_transform(
    df["text_cleaned"]
)

similarity_graph = radius_neighbors_graph(
    similarity_vectors,
    radius=0.10,
    mode="connectivity",
    metric="cosine",
    include_self=True,
    n_jobs=-1
)

similarity_graph = similarity_graph.maximum(similarity_graph.T)

number_of_clusters, cluster_ids = connected_components(
    similarity_graph,
    directed=False
)

df["cluster_id"] = cluster_ids

print("Similarity clusters:", number_of_clusters)

# 3. Split by similarity cluster
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(
        df["text_cleaned"],
        df["binary_label"],
        groups=df["cluster_id"]
    )
)

X_train = df.iloc[train_indices]["text_cleaned"]
X_test = df.iloc[test_indices]["text_cleaned"]

y_train = df.iloc[train_indices]["binary_label"]
y_test = df.iloc[test_indices]["binary_label"]

train_groups = set(df.iloc[train_indices]["cluster_id"])
test_groups = set(df.iloc[test_indices]["cluster_id"])

print("\nTRAINING AND TESTING SPLIT")
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))
print("Training clusters:", len(train_groups))
print("Testing clusters:", len(test_groups))
print("Shared clusters:", len(train_groups.intersection(test_groups)))

# 4. Train a new baseline model
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=30000,
            min_df=2,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

print("\nTraining cluster-aware model...")
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)

print("\nCLUSTER-AWARE MODEL EVALUATION")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Scam"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 6. Save separately
joblib.dump(
    model,
    BASE_DIR / "vaaniguard_cluster_aware_baseline.pkl"
)

print("\nSaved: vaaniguard_cluster_aware_baseline.pkl")
print("CLUSTER-AWARE TRAINING COMPLETED")
