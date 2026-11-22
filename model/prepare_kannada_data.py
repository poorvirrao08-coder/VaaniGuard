
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# Load dataset (skip the extra title row)
df = pd.read_csv(
    BASE_DIR / "vaaniguard_kannada_mixed_pilot_expanded_180.csv",
    skiprows=1
)

# Standardize language format names
language_mapping = {
    "kannada_script": "kannada_script",
    "romanized_kannada": "romanized_kannada",
    "kannada_english_mixed": "kannada_english_mixed",
    "romanized_kannada_english": "romanized_kannada"
}

df["language_format"] = (
    df["language_format"]
    .str.strip()
    .str.lower()
    .replace(language_mapping)
)

# Standardize labels
df["label"] = df["label"].str.strip().str.lower()

print("\nDATASET SHAPE:")
print(df.shape)

print("\nLANGUAGE DISTRIBUTION:")
print(df["language_format"].value_counts())

print("\nLABEL DISTRIBUTION:")
print(df["label"].value_counts())

print("\nSPLIT DISTRIBUTION:")
print(df["split"].value_counts())

print("\nMISSING VALUES:")
print(df.isnull().sum())

# Save standardized dataset
df.to_csv(
    BASE_DIR / "vaaniguard_kannada_prepared.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nPrepared dataset saved successfully!")
