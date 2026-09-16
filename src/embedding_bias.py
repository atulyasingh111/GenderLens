import numpy as np
import gensim.downloader as api


# Name of the pretrained Word2Vec model
MODEL_NAME = "word2vec-google-news-300"


# Load the pretrained Word2Vec model
print("Loading Word2Vec model...")

model = api.load(MODEL_NAME)

print("Model loaded successfully.")


# Gender attribute word sets
MALE_WORDS = ["he", "him", "man", "boy"]
FEMALE_WORDS = ["she", "her", "woman", "girl"]


def association_score(word):
    """
    Calculate the word-level gender association score.

    Score =
    average similarity with male attribute words
    -
    average similarity with female attribute words

    Interpretation:
        score > 0  -> Male-associated
        score < 0  -> Female-associated
        score ≈ 0  -> Near-balanced
    """

    male_similarities = [
        model.similarity(word, attribute)
        for attribute in MALE_WORDS
    ]

    female_similarities = [
        model.similarity(word, attribute)
        for attribute in FEMALE_WORDS
    ]

    male_average = np.mean(male_similarities)
    female_average = np.mean(female_similarities)

    gender_association_score = male_average - female_average

    return male_average, female_average, gender_association_score


def get_association_label(score, threshold=0.01):
    """
    Convert the numerical association score into a neutral label.
    """

    if score > threshold:
        return "Male-associated"

    elif score < -threshold:
        return "Female-associated"

    else:
        return "Near-balanced"


def analyze_word(word):
    """
    Calculate similarities, association score,
    and association label for one word.
    """

    male_average, female_average, gender_association_score = (
        association_score(word)
    )

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


# Test the analysis with one occupation
if __name__ == "__main__":

    test_word = "engineer"

    result = analyze_word(test_word)

    print("\nGenderLens Test Result")
    print("----------------------")
    print("Word:", result["word"])
    print(
        "Average similarity with male attributes:",
        result["male_similarity"]
    )
    print(
        "Average similarity with female attributes:",
        result["female_similarity"]
    )
    print(
        "Gender association score:",
        result["gender_association_score"]
    )
    print(
        "Association:",
        result["association_label"]
    )
