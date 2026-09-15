import numpy as np
from gensim.models import KeyedVectors


# Path to the pretrained Word2Vec model
MODEL_PATH = "models/GoogleNews-vectors-negative300.bin"


# Load the pretrained Word2Vec model
model = KeyedVectors.load_word2vec_format(
    MODEL_PATH,
    binary=True
)


# Gender attribute word sets
MALE_WORDS = ["he", "him", "man", "boy"]
FEMALE_WORDS = ["she", "her", "woman", "girl"]


def association_score(word):
    """
    Calculate the word-level gender association score.

    The score is calculated as:

    average similarity with male words
    -
    average similarity with female words

    Interpretation:
        score > 0  -> Male-associated
        score < 0  -> Female-associated
        score ≈ 0  -> Near-balanced
    """

    # Similarity between the occupation and male attribute words
    male_similarities = [
        model.similarity(word, attribute)
        for attribute in MALE_WORDS
    ]

    # Similarity between the occupation and female attribute words
    female_similarities = [
        model.similarity(word, attribute)
        for attribute in FEMALE_WORDS
    ]

    # Average similarity for each attribute group
    male_average = np.mean(male_similarities)
    female_average = np.mean(female_similarities)

    # Difference between the two averages
    gender_association_score = male_average - female_average

    return male_average, female_average, gender_association_score


def get_association_label(score, threshold=0.01):
    """
    Convert the numerical association score into a neutral label.

    A small threshold is used because a score extremely close to
    zero should be treated as approximately balanced.
    """

    if score > threshold:
        return "Male-associated"

    elif score < -threshold:
        return "Female-associated"

    else:
        return "Near-balanced"


def analyze_word(word):
    """
    Calculate similarities, association score, and label for one word.
    """

    male_average, female_average, gender_association_score = association_score(word)

    association_label = get_association_label(
        gender_association_score
    )

    return {
        "word": word,
        "male_similarity": float(male_average),
        "female_similarity": float(female_average),
        "gender_association_score": float(gender_association_score),
        "association_label": association_label
    }


# Example usage
if __name__ == "__main__":

    test_word = "engineer"

    result = analyze_word(test_word)

    print("Word:", result["word"])
    print("Average similarity with male attributes:",
          result["male_similarity"])
    print("Average similarity with female attributes:",
          result["female_similarity"])
    print("Gender association score:",
          result["gender_association_score"])
    print("Association:", result["association_label"])
