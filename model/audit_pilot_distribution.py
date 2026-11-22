
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "vaaniguard_kannada_prepared.csv")

train = df[df["split"] == "synthetic_expansion_pilot"]
test = df[df["split"] == "evaluation_pilot"]

shared = sorted(
    set(train["scenario"]) & set(test["scenario"])
)

print("\nSHARED SCENARIO INVESTIGATION")

for scenario in shared:
    print("\nScenario:", scenario)

    print("\nExpansion examples:")
    print(
        train[train["scenario"] == scenario][
            ["id", "language_format", "label", "text"]
        ].to_string(index=False)
    )

    print("\nEvaluation examples:")
    print(
        test[test["scenario"] == scenario][
            ["id", "language_format", "label", "text"]
        ].to_string(index=False)
    )

print("\nINVESTIGATION COMPLETED")
