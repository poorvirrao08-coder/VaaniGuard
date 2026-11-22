
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "vaaniguard_cluster_aware_baseline.pkl")

vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

features = vectorizer.get_feature_names_out()
weights = classifier.coef_[0]

ranked = sorted(
    zip(features, weights),
    key=lambda item: item[1]
)

print("\nTOP 30 LEGITIMATE-ASSOCIATED FEATURES")
for word, score in ranked[:30]:
    print(f"{word:35s} {score:.4f}")

print("\nTOP 30 SCAM-ASSOCIATED FEATURES")
for word, score in ranked[-30:][::-1]:
    print(f"{word:35s} {score:.4f}")

print("\nMODEL FEATURE INSPECTION COMPLETED")
