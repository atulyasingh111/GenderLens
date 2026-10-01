import json
import pandas as pd


BEFORE_PATH = "results/before_debiasing.json"
AFTER_PATH = "results/after_debiasing.json"


with open(BEFORE_PATH) as file:
    before = json.load(file)

with open(AFTER_PATH) as file:
    after = json.load(file)


before_df = pd.DataFrame(before["occupations"])
after_df = pd.DataFrame(after["occupations"])


comparison = before_df[
    ["word", "gender_association_score", "association_label"]
].merge(
    after_df[
        ["word", "gender_association_score", "association_label"]
    ],
    on="word",
    suffixes=("_before", "_after")
)


comparison["absolute_score_before"] = (
    comparison["gender_association_score_before"].abs()
)

comparison["absolute_score_after"] = (
    comparison["gender_association_score_after"].abs()
)

comparison["absolute_change"] = (
    comparison["absolute_score_before"]
    - comparison["absolute_score_after"]
)


print("\nBEFORE vs AFTER DEBIASING\n")

print(
    comparison[
        [
            "word",
            "gender_association_score_before",
            "gender_association_score_after",
            "absolute_change"
        ]
    ].to_string(index=False)
)


mean_before = comparison["absolute_score_before"].mean()
mean_after = comparison["absolute_score_after"].mean()

print("\nSummary:")
print(f"Mean absolute association before: {mean_before:.4f}")
print(f"Mean absolute association after:  {mean_after:.4f}")

if mean_before != 0:
    reduction = (
        (mean_before - mean_after)
        / mean_before
        * 100
    )
    print(f"Mean absolute association change: {reduction:.2f}%")

