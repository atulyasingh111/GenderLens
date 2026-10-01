import numpy as np
import gensim.downloader as api


MODEL_NAME = "word2vec-google-news-300"

GENDER_PAIRS = [
    ("he", "she"),
    ("him", "her"),
    ("man", "woman"),
    ("boy", "girl"),
]


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

    # Normalize the direction
    gender_direction = gender_direction / np.linalg.norm(gender_direction)

    return gender_direction


gender_direction = create_gender_direction(model, GENDER_PAIRS)

print("\nGender direction created.")
print("Dimensions:", gender_direction.shape)
print("First 5 values:", gender_direction[:5])
def gender_projection(word, model, gender_direction):
    vector = model[word]

    projection = np.dot(vector, gender_direction)

    return projection


test_words = [
    "he",
    "she",
    "man",
    "woman",
    "engineer",
    "nurse",
    "doctor",
]

print("\nGender direction projections:")

for word in test_words:
    projection = gender_projection(word, model, gender_direction)
    print(f"{word}: {projection:.4f}")
def debias_word(word, model, gender_direction):
    vector = model[word].copy()

    projection = np.dot(vector, gender_direction) * gender_direction

    debiased_vector = vector - projection

    return debiased_vector


print("\nTesting debiasing:")

for word in ["engineer", "nurse", "doctor", "teacher", "manager"]:
    original = model[word]
    debiased = debias_word(word, model, gender_direction)

    original_projection = np.dot(original, gender_direction)
    debiased_projection = np.dot(debiased, gender_direction)

    print(
        f"{word}: "
        f"before={original_projection:.4f}, "
        f"after={debiased_projection:.4f}"
    )
