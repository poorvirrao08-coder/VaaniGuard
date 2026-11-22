
from pathlib import Path

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load prepared dataset
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(
    BASE_DIR / "vaaniguard_kannada_prepared.csv",
    encoding="utf-8-sig"
)

# 2. Clean required columns
df["text"] = df["text"].fillna("").astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["split"] = df["split"].astype(str).str.strip()

df = df[df["text"] != ""].copy()

# 3. Separate training and evaluation data
train_df = df[
    df["split"] == "synthetic_expansion_pilot"
].copy()

test_df = df[
    df["split"] == "evaluation_pilot"
].copy()

print("Training records:", len(train_df))
print("Evaluation records:", len(test_df))

print("\nTraining label distribution:")
print(train_df["label"].value_counts())

print("\nEvaluation label distribution:")
print(test_df["label"].value_counts())

# 4. Prepare input and output
X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]

# 5. Build the model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        max_features=20000,
        sublinear_tf=True
    )),

    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# 6. Train the model
print("\nTraining Kannada-English model...")

model.fit(X_train, y_train)

print("Training completed!")

# 7. Evaluate the model
predictions = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=["legitimate", "scam"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["legitimate", "scam"]
))

# 8. Evaluate individual language formats
print("\nLanguage-wise evaluation:")

for language in test_df["language_format"].unique():

    mask = test_df["language_format"] == language

    print(f"\n{language}")
    print(classification_report(
        y_test[mask],
        predictions[mask],
        labels=["legitimate", "scam"],
        zero_division=0
    ))

# 9. Save trained model
joblib.dump(model, BASE_DIR / "vaaniguard_kannada_baseline.pkl")

print("\nModel saved successfully!")
print("File: vaaniguard_kannada_baseline.pkl")

