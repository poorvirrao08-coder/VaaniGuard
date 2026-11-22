
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "vaaniguard_kannada_prepared.csv")

print("\nKANNADA PILOT DATASET QUALITY AUDIT")

print("\n1. LANGUAGE DISTRIBUTION")
print(df["language_format"].value_counts())

print("\n2. LABEL DISTRIBUTION")
print(df["label"].value_counts())

print("\n3. LANGUAGE AND LABEL DISTRIBUTION")
print(pd.crosstab(df["language_format"], df["label"]))

print("\n4. REVIEW STATUS")
print(df["review_status"].value_counts())

print("\n5. DATA ORIGIN")
print(df["origin"].value_counts())

print("\n6. DATA SPLIT")
print(df["split"].value_counts())

print("\n7. DUPLICATE TEXTS")
print("Duplicates:", df["text"].duplicated().sum())

print("\n8. EXAMPLES BY LANGUAGE AND LABEL")

for language in df["language_format"].unique():
    for label in ["scam", "legitimate"]:
        subset = df[
            (df["language_format"] == language) &
            (df["label"] == label)
        ]

        print(f"\n{language} | {label}")

        for text in subset["text"].head(5):
            print("-", text)

print("\nPILOT QUALITY AUDIT COMPLETED")
