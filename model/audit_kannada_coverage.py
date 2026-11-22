
import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "vaaniguard_cleaned_sentinel_dataset.csv")
df = df[["text_cleaned", "binary_label"]].dropna().copy()

df["text_cleaned"] = df["text_cleaned"].astype(str)

# Detect Kannada script
kannada_pattern = re.compile(r"[\u0C80-\u0CFF]")

df["has_kannada_script"] = df["text_cleaned"].apply(
    lambda text: bool(kannada_pattern.search(text))
)

# Basic indicators of Romanized Kannada.
# This is only a preliminary check, not reliable language identification.
kanglish_words = {
    "nimma", "nimage", "nimge", "madi", "maadi",
    "beku", "beda", "illa", "ide", "agide",
    "barutte", "hogbeda", "yenu", "enu", "yaake",
    "dayavittu", "sahaya", "kodi", "haudu"
}

def has_kanglish(text):
    words = set(re.findall(r"[a-z]+", text.lower()))
    return bool(words.intersection(kanglish_words))

df["possible_kanglish"] = df["text_cleaned"].apply(has_kanglish)

print("\nKANNADA LANGUAGE COVERAGE AUDIT")
print("Total messages:", len(df))

print(
    "Messages containing Kannada script:",
    df["has_kannada_script"].sum()
)

print(
    "Messages with possible Romanized Kannada:",
    df["possible_kanglish"].sum()
)

print(
    "Messages with neither detected:",
    (~df["has_kannada_script"] & ~df["possible_kanglish"]).sum()
)

print("\nKANNADA SCRIPT EXAMPLES")
for text in df.loc[df["has_kannada_script"], "text_cleaned"].head(10):
    print("-", text)

print("\nPOSSIBLE KANGLISH EXAMPLES")
for text in df.loc[df["possible_kanglish"], "text_cleaned"].head(10):
    print("-", text)

print("\nAUDIT COMPLETED")
