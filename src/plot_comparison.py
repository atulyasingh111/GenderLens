import json
import pandas as pd
import matplotlib.pyplot as plt


BEFORE_PATH = "results/before_debiasing.json"
AFTER_PATH = "results/after_debiasing.json"


with open(BEFORE_PATH) as file:
    before = json.load(file)

with open(AFTER_PATH) as file:
    after = json.load(file)


before_df = pd.DataFrame(before["occupations"])
after_df = pd.DataFrame(after["occupations"])


comparison = before_df[
    ["word", "gender_association_score"]
].merge(
    after_df[
        ["word", "gender_association_score"]
    ],
    on="word",
    suffixes=("_before", "_after")
)


comparison = comparison.sort_values(
    "gender_association_score_before"
)


plt.figure(figsize=(12, 8))

plt.barh(
    comparison["word"],
    comparison["gender_association_score_before"],
    alpha=0.7,
    label="Before debiasing"
)

plt.barh(
    comparison["word"],
    comparison["gender_association_score_after"],
    alpha=0.7,
    label="After debiasing"
)

plt.axvline(
    0,
    linewidth=1
)

plt.xlabel("Gender association score")
plt.ylabel("Occupation")
plt.title("GenderLens: Before vs After Debiasing")
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/before_after_comparison.png",
    dpi=300
)

plt.show()

print(
    "\nGraph saved to "
    "results/before_after_comparison.png"
)
