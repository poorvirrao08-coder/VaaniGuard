
from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("\nVAANIGUARD SCENARIO-HELD-OUT MODEL EVALUATION")
print("=" * 60)

# Load the saved scenario-held-out dataset
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "vaaniguard_scenario_heldout_split.csv")

train = df[df["split"] == "training"].copy()
test = df[df["split"] == "evaluation"].copy()

# Check dataset
print("\nDATASET")
print("Training messages:", len(train))
print("Evaluation messages:", len(test))

# Check scenario separation
shared = set(train["scenario"]) & set(test["scenario"])
print("Shared scenarios:", len(shared))

if shared:
    raise ValueError("Scenario overlap detected. Evaluation stopped.")

# Convert labels to binary values
label_map = {
    "legitimate": 0,
    "scam": 1
}

train["target"] = train["label"].map(label_map)
test["target"] = test["label"].map(label_map)

if train["target"].isna().any() or test["target"].isna().any():
    raise ValueError("Unexpected label found in dataset.")

X_train = train["text"].astype(str)
y_train = train["target"]

X_test = test["text"].astype(str)
y_test = test["target"]

# Build experimental ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1,
            max_features=20000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])

# Train only on training data
print("\nTraining experimental model...")
model.fit(X_train, y_train)

# Predict unseen evaluation messages
predictions = model.predict(X_test)

# Calculate evaluation metrics
print("\nEVALUATION RESULTS")
print("-" * 40)

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

print(
    "Precision:",
    round(precision_score(y_test, predictions, zero_division=0), 4)
)

print(
    "Recall:",
    round(recall_score(y_test, predictions, zero_division=0), 4)
)

print(
    "F1-score:",
    round(f1_score(y_test, predictions, zero_division=0), 4)
)

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["legitimate", "scam"],
        zero_division=0
    )
)

print("\nCONFUSION MATRIX")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )
)

# Evaluate each language format separately
print("\nLANGUAGE-WISE RESULTS")
print("-" * 40)

results = []

for language in sorted(test["language_format"].unique()):

    indices = test["language_format"] == language

    actual = y_test[indices]
    predicted = predictions[indices]

    accuracy = accuracy_score(actual, predicted)
    f1 = f1_score(actual, predicted, zero_division=0)

    results.append({
        "language_format": language,
        "messages": int(indices.sum()),
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4)
    })

print(pd.DataFrame(results).to_string(index=False))

# Save predictions separately for analysis
output = test[
    ["id", "language_format", "scenario", "label", "text"]
].copy()

output["predicted_label"] = [
    "scam" if prediction == 1 else "legitimate"
    for prediction in predictions
]

output["correct"] = (
    output["label"] == output["predicted_label"]
)

output.to_csv(
    BASE_DIR / "vaaniguard_scenario_evaluation_predictions.csv",
    index=False
)

print("\nSaved predictions:")
print("vaaniguard_scenario_evaluation_predictions.csv")

print("\nEXPERIMENT COMPLETED")
