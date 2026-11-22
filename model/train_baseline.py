
import re
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "vaaniguard_cleaned_sentinel_dataset.csv")

df = df[["text_cleaned", "binary_label"]].dropna().copy()
df["text_cleaned"] = df["text_cleaned"].astype(str).str.strip()
df = df[df["text_cleaned"] != ""]

df["binary_label"] = df["binary_label"].astype(int)

# 2. Remove conflicting duplicate messages
conflicts = df.groupby("text_cleaned")["binary_label"].nunique()
ambiguous_texts = conflicts[conflicts > 1].index

print("Conflicting duplicate messages:", len(ambiguous_texts))

df = df[~df["text_cleaned"].isin(ambiguous_texts)]
df = df.drop_duplicates(subset="text_cleaned").reset_index(drop=True)

print("Dataset after cleaning:", df.shape)

# 3. Create approximate message-template groups
def normalize_template(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\b[\w.-]+\.com\b", " URL ", text)
    text = re.sub(r"\b\d+\b", " NUMBER ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

groups = df["text_cleaned"].apply(normalize_template)

# 4. Group-aware train-test split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, df["binary_label"], groups=groups)
)

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

X_train = train_df["text_cleaned"]
y_train = train_df["binary_label"]

X_test = test_df["text_cleaned"]
y_test = test_df["binary_label"]

print("\nGROUP-AWARE DATA SPLIT")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training groups:", groups.iloc[train_idx].nunique())
print("Testing groups:", groups.iloc[test_idx].nunique())

# Check class distribution
print("\nTRAINING CLASS DISTRIBUTION")
print(y_train.value_counts())

print("\nTEST CLASS DISTRIBUTION")
print(y_test.value_counts())

# 5. TF-IDF + Logistic Regression
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

# 6. Train model
print("\nTraining group-aware baseline model...")
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)

print("\n========== GROUP-AWARE EVALUATION ==========")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Legitimate", "Scam"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# 8. Save model and split information
joblib.dump(model, BASE_DIR / "vaaniguard_baseline.pkl")

print("\nModel saved as vaaniguard_baseline.pkl")
print("Training and evaluation completed.")

