import numpy as np
import pandas as pd
import gensim.downloader as api


MODEL_NAME = "word2vec-google-news-300"

GENDER_PAIRS = [
    ("he", "she"),
    ("him", "her"),
    ("man", "woman"),
    ("boy", "girl"),
]

OCCUPATIONS_PATH = "data/occupations.csv"


print("Loading Word2Vec model...")
model = api.load(MODEL_NAME)
print("Model loaded.")


def create_gender_direction(model, gender_pairs):
    directions = []

    for male_word, female_word in gender_pairs:
        male_vector = model[male_word]
        female_vector = model[female_word]

        direction = male_vector - female_vector
        directions.append(direction)

    gender_direction = np.mean(directions, axis=0)

    gender_direction = (
        gender_direction / np.linalg.norm(gender_direction)
    )

    return gender_direction


def debias_word(word, model, gender_direction):
    vector = model[word].copy()

    projection = (
        np.dot(vector, gender_direction)
        * gender_direction
    )

    debiased_vector = vector - projection

    return debiased_vector


gender_direction = create_gender_direction(
    model,
    GENDER_PAIRS
)

print("\nGender direction created.")
print("Dimensions:", gender_direction.shape)


occupations = pd.read_csv(OCCUPATIONS_PATH)

print("\nApplying debiasing to occupations...")

debiased_vectors = {}

for word in occupations["occupation"]:
    word = word.strip()

    debiased_vectors[word] = debias_word(
        word,
        model,
        gender_direction
    )

    print(f"Debiased: {word}")


print("\nDebiasing complete.")
print("Total occupations processed:", len(debiased_vectors))
def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


def association_score_debiased(word, debiased_vector):
    male_similarities = []
    female_similarities = []

    for male_word, female_word in GENDER_PAIRS:
        male_similarities.append(
            cosine_similarity(
                debiased_vector,
                model[male_word]
            )
        )

        female_similarities.append(
            cosine_similarity(
                debiased_vector,
                model[female_word]
            )
        )

    male_similarity = np.mean(male_similarities)
    female_similarity = np.mean(female_similarities)

    score = male_similarity - female_similarity

    return male_similarity, female_similarity, score


print("\nAfter-debiasing association scores:")

import json

AFTER_RESULTS_PATH = "results/after_debiasing.json"

after_results = {
    "model": MODEL_NAME,
    "male_attributes": ["he", "him", "man", "boy"],
    "female_attributes": ["she", "her", "woman", "girl"],
    "near_balanced_threshold": 0.01,
    "method": "Gender-direction neutralization",
    "occupations": []
}

print("\nAfter-debiasing association scores:")

for word, debiased_vector in debiased_vectors.items():

    male_similarities = []
    female_similarities = []

    for male_word, female_word in GENDER_PAIRS:

        male_similarities.append(
            cosine_similarity(
                debiased_vector,
                model[male_word]
            )
        )

        female_similarities.append(
            cosine_similarity(
                debiased_vector,
                model[female_word]
            )
        )

    male_similarity = np.mean(male_similarities)
    female_similarity = np.mean(female_similarities)

    score = male_similarity - female_similarity

    if score > 0.01:
        label = "Male-associated"
    elif score < -0.01:
        label = "Female-associated"
    else:
        label = "Near-balanced"

    result = {
        "word": word,
        "male_similarity": float(male_similarity),
        "female_similarity": float(female_similarity),
        "gender_association_score": float(score),
        "association_label": label
    }

    after_results["occupations"].append(result)

    print(
        f"{word}: "
        f"male={male_similarity:.4f}, "
        f"female={female_similarity:.4f}, "
        f"score={score:.4f}"
    )


with open(AFTER_RESULTS_PATH, "w") as file:
    json.dump(after_results, file, indent=4)

print(f"\nResults saved to {AFTER_RESULTS_PATH}")
