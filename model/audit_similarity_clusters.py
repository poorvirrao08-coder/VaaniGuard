
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse.csgraph import connected_components

# Load and clean dataset
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "vaaniguard_cleaned_sentinel_dataset.csv")
df = df[["text_cleaned", "binary_label"]].dropna().copy()

df["text_cleaned"] = df["text_cleaned"].astype(str).str.strip()
df = df[df["text_cleaned"] != ""]
df = df.drop_duplicates(subset="text_cleaned").reset_index(drop=True)

# Character-level TF-IDF
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=100000
)

vectors = vectorizer.fit_transform(df["text_cleaned"])

# Connect messages with cosine similarity >= 0.90
print("Building similarity graph...")

graph = radius_neighbors_graph(
    vectors,
    radius=0.10,
    mode="connectivity",
    metric="cosine",
    include_self=True,
    n_jobs=-1
)

graph = graph.maximum(graph.T)

# Identify connected similarity groups
number_of_clusters, cluster_ids = connected_components(
    graph,
    directed=False
)

df["cluster_id"] = cluster_ids

cluster_sizes = df["cluster_id"].value_counts()

print("\nSIMILARITY CLUSTER AUDIT")
print("Total messages:", len(df))
print("Similarity clusters:", number_of_clusters)
print("Largest cluster:", cluster_sizes.max())
print("Singleton messages:", (cluster_sizes == 1).sum())
print("Clusters with multiple messages:", (cluster_sizes > 1).sum())

# Check clusters containing both labels
mixed_clusters = (
    df.groupby("cluster_id")["binary_label"].nunique() > 1
).sum()

print("Clusters containing both labels:", mixed_clusters)

print("\nTOP 10 LARGEST CLUSTERS")

for cluster_id, size in cluster_sizes.head(10).items():
    subset = df[df["cluster_id"] == cluster_id]

    print("\nCluster:", cluster_id)
    print("Messages:", size)
    print("Labels:", subset["binary_label"].value_counts().to_dict())
    print("Example:", subset.iloc[0]["text_cleaned"][:250])

print("\nSIMILARITY CLUSTER AUDIT COMPLETED")
