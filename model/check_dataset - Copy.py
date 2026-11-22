
from pathlib import Path

import pandas as pd

# Load the Sentinel dataset
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "vaaniguard_cleaned_sentinel_dataset.csv")

# Display dataset information
print("DATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nFIRST 5 RECORDS:")
print(df.head())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nLABEL DISTRIBUTION:")
for col in ["label", "binary_label", "split"]:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts())
