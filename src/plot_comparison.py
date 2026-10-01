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

y = range(len(comparison))
height = 0.35

plt.barh(
    [i - height / 2 for i in y],
    comparison["gender_association_score_before"],
    height=height,
    alpha=0.8,
    label="Before debiasing"
)

plt.barh(
    [i + height / 2 for i in y],
    comparison["gender_association_score_after"],
    height=height,
    alpha=0.8,
    label="After debiasing"
)

plt.yticks(y, comparison["word"])
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
