import json
import numpy as np
import gensim.downloader as api

MODEL_NAME = "word2vec-google-news-300"

GENDER_PAIRS = [
    ("he", "she"),
    ("him", "her"),
    ("man", "woman"),
    ("boy", "girl"),
]

SEMANTIC_PAIRS = {
    "engineer": ["technology", "machine", "design"],
    "doctor": ["hospital", "medicine", "patient"],
    "nurse": ["hospital", "patient", "medicine"],
    "teacher": ["school", "student", "education"],
    "lawyer": ["court", "legal", "law"],
    "banker": ["bank", "finance", "money"],
    "scientist": ["research", "laboratory", "experiment"],
    "programmer": ["software", "computer", "coding"],
    "accountant": ["finance", "accounting", "numbers"],
    "manager": ["business", "team", "company"],
    "secretary": ["office", "administration", "assistant"],
    "receptionist": ["office", "hotel", "customer"],
    "mechanic": ["engine", "car", "repair"],
    "pilot": ["aircraft", "aviation", "flight"],
    "chef": ["restaurant", "food", "cooking"],
    "designer": ["design", "creative", "art"],
    "artist": ["art", "creative", "painting"],
    "journalist": ["news", "media", "reporter"],
    "investor": ["finance", "market", "stocks"],
    "ceo": ["company", "business", "executive"],
    "boss": ["manager", "company", "work"],
    "leader": ["team", "management", "organization"],
}

print("Loading Word2Vec model...")
model = api.load(MODEL_NAME)
print("Model loaded.")


def create_gender_direction(model, gender_pairs):
    directions = []

    for male_word, female_word in gender_pairs:
        male_vector = model[male_word]
        female_vector = model[female_word]

        difference = male_vector - female_vector
        directions.append(difference)

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


def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )


gender_direction = create_gender_direction(
    model,
    GENDER_PAIRS
)

print("\nGender direction created.")

results = []

print("\nSemantic preservation analysis:")

for occupation, related_words in SEMANTIC_PAIRS.items():

    original_vector = model[occupation]

    debiased_vector = debias_word(
        occupation,
        model,
        gender_direction
    )

    for related_word in related_words:

        related_vector = model[related_word]

        similarity_before = cosine_similarity(
            original_vector,
            related_vector
        )

        similarity_after = cosine_similarity(
            debiased_vector,
            related_vector
        )

        change = similarity_after - similarity_before

        results.append({
            "occupation": occupation,
            "related_word": related_word,
            "similarity_before": float(similarity_before),
            "similarity_after": float(similarity_after),
            "change": float(change)
        })

        print(
            f"{occupation} -> {related_word}: "
            f"before={similarity_before:.4f}, "
            f"after={similarity_after:.4f}, "
            f"change={change:.4f}"
        )


absolute_changes = [
    abs(result["change"])
    for result in results
]

mean_absolute_change = np.mean(absolute_changes)

print("\nSummary")
print("-------")
print(f"Semantic relationships tested: {len(results)}")
print(
    f"Mean absolute similarity change: "
    f"{mean_absolute_change:.4f}"
)

output_path = "results/semantic_preservation.json"

with open(output_path, "w") as file:
    json.dump(
        {
            "model": MODEL_NAME,
            "method": "Gender-direction neutralization",
            "semantic_relationships_tested": len(results),
            "mean_absolute_similarity_change": float(
                mean_absolute_change
            ),
            "results": results
        },
        file,
        indent=4
    )

print(f"\nResults saved to {output_path}")
