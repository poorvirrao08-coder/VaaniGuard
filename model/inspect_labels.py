from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "vaaniguard_cleaned_sentinel_dataset.csv")

print("SOURCE LABEL DISTRIBUTION:")
print(df["source_label"].value_counts())

print("\nINFERRED LABEL MEANINGS:")
print(df["source_label_meaning_inferred"].value_counts())

print("\nBINARY LABEL MAPPING:")
print(pd.crosstab(
    df["source_label_meaning_inferred"],
    df["binary_label"]
))

print("\nSCAM CATEGORY DISTRIBUTION:")
print(df["scam_category_candidate"].value_counts())

print("\nSAMPLE TEXT WITH LABELS:")
print(df[[
    "text_cleaned",
    "source_label_meaning_inferred",
    "binary_label"
]].sample(10, random_state=42).to_string(index=False))
