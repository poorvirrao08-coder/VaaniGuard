
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "vaaniguard_kannada_prepared.csv")

print("\nSCENARIO-HELD-OUT SPLIT AUDIT")
print("=" * 55)

X = df["text"]
y = df["label"]
groups = df["scenario"]

languages = sorted(df["language_format"].unique())
labels = sorted(df["label"].unique())

best_split = None
best_score = float("inf")

for seed in range(100, 1100):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.33,
        random_state=seed
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    if set(train["scenario"]) & set(test["scenario"]):
        continue

    if set(train["label"]) != set(labels):
        continue

    if set(test["label"]) != set(labels):
        continue

    if set(train["language_format"]) != set(languages):
        continue

    if set(test["language_format"]) != set(languages):
        continue

    train_language = train["language_format"].value_counts(normalize=True)
    test_language = test["language_format"].value_counts(normalize=True)

    train_label = train["label"].value_counts(normalize=True)
    test_label = test["label"].value_counts(normalize=True)

    language_score = sum(
        abs(train_language.get(lang, 0) - test_language.get(lang, 0))
        for lang in languages
    )

    label_score = sum(
        abs(train_label.get(label, 0) - test_label.get(label, 0))
        for label in labels
    )

    score = language_score + label_score

    if score < best_score:
        best_score = score
        best_split = (train.copy(), test.copy(), seed)

if best_split is None:
    print("\nNo suitable split found.")

else:
    train, test, seed = best_split

    print("\nBEST CANDIDATE FOUND")
    print("Random seed:", seed)
    print("Balance score:", round(best_score, 4))

    print("\nTRAINING SET")
    print("Messages:", len(train))
    print("Scenarios:", train["scenario"].nunique())

    print("\nEVALUATION SET")
    print("Messages:", len(test))
    print("Scenarios:", test["scenario"].nunique())

    print("\nEVALUATION LANGUAGE AND LABEL DISTRIBUTION")
    print(pd.crosstab(test["language_format"], test["label"]))

    print("\nTRAINING LABEL DISTRIBUTION")
    print(train["label"].value_counts())

    print("\nEVALUATION LABEL DISTRIBUTION")
    print(test["label"].value_counts())

    shared = set(train["scenario"]) & set(test["scenario"])
    print("\nShared scenarios:", len(shared))

    print("\nEVALUATION SCENARIOS")
    print(sorted(test["scenario"].unique()))

    output = pd.concat([
        train.assign(split="training"),
        test.assign(split="evaluation")
    ])

    output.to_csv(
        BASE_DIR / "vaaniguard_scenario_heldout_split.csv",
        index=False
    )

    print("\nSaved: vaaniguard_scenario_heldout_split.csv")

print("\nSCENARIO SPLIT AUDIT COMPLETED")
