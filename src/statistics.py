import json
import numpy as np

BEFORE_PATH = "results/before_debiasing.json"
AFTER_PATH = "results/after_debiasing.json"

N_PERMUTATIONS = 10000
RANDOM_SEED = 42


def load_scores(path):
    with open(path, "r") as file:
        data = json.load(file)

    return {
        item["word"]: item["gender_association_score"]
        for item in data["occupations"]
    }


before = load_scores(BEFORE_PATH)
after = load_scores(AFTER_PATH)

words = list(before.keys())

before_abs = np.array([abs(before[word]) for word in words])
after_abs = np.array([abs(after[word]) for word in words])

observed_difference = np.mean(before_abs - after_abs)

rng = np.random.default_rng(RANDOM_SEED)

permuted_differences = []

for _ in range(N_PERMUTATIONS):
    signs = rng.choice([-1, 1], size=len(words))
    difference = np.mean(
        signs * (before_abs - after_abs)
    )
    permuted_differences.append(difference)

permuted_differences = np.array(permuted_differences)

p_value = (
    np.sum(permuted_differences >= observed_difference)
    / N_PERMUTATIONS
)

print("Statistical Validation")
print("----------------------")
print(f"Occupations analysed: {len(words)}")
print(f"Observed mean reduction: {observed_difference:.4f}")
print(f"Permutation tests: {N_PERMUTATIONS}")
print(f"p-value: {p_value:.4f}")
