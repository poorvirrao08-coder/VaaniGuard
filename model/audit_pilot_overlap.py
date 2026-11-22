
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "vaaniguard_kannada_prepared.csv")

train = df[df["split"] == "synthetic_expansion_pilot"].copy()
test = df[df["split"] == "evaluation_pilot"].copy()

print("\nPILOT SPLIT OVERLAP AUDIT")
print("Training examples:", len(train))
print("Evaluation examples:", len(test))

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=1
)

all_texts = pd.concat(
    [train["text"], test["text"]],
    ignore_index=True
)

vectors = vectorizer.fit_transform(all_texts)

train_vectors = vectors[:len(train)]
test_vectors = vectors[len(train):]

similarities = cosine_similarity(
    test_vectors,
    train_vectors
)

max_similarities = similarities.max(axis=1)
nearest_indices = similarities.argmax(axis=1)

print("\nSIMILARITY RESULTS")
print("Average maximum similarity:", round(max_similarities.mean(), 4))
print("Similarity >= 0.80:", (max_similarities >= 0.80).sum())
print("Similarity >= 0.90:", (max_similarities >= 0.90).sum())
print("Similarity >= 0.95:", (max_similarities >= 0.95).sum())

print("\nTOP 10 MOST SIMILAR PAIRS")

for i in max_similarities.argsort()[-10:][::-1]:
    j = nearest_indices[i]

    print("\nSimilarity:", round(max_similarities[i], 4))
    print("Evaluation:", test.iloc[i]["text"])
    print("Training:", train.iloc[j]["text"])
    print("Evaluation label:", test.iloc[i]["label"])
    print("Training label:", train.iloc[j]["label"])

print("\nPILOT OVERLAP AUDIT COMPLETED")
