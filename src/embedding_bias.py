import numpy as np
import pandas as pd
import gensim.downloader as api


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "word2vec-google-news-300"

OCCUPATIONS_PATH = "data/occupations.csv"
REFERENCE_PATH = "data/reference_words.csv"
EXPLORATORY_PATH = "data/exploratory_terms.csv"

MALE_WORDS = ["he", "him", "man", "boy"]
FEMALE_WORDS = ["she", "her", "woman", "girl"]

NEAR_BALANCED_THRESHOLD = 0.01


# --------------------------------------------------
# Load pretrained Word2Vec model
# --------------------------------------------------

print("Loading Word2Vec model...")
model = api.load(MODEL_NAME)
print("Model loaded successfully.")


# --------------------------------------------------
# Calculate gender association score
# --------------------------------------------------

def association_score(word):
    """
    Calculate the word-level gender association score.

    Score =
    average similarity with male attribute words
    minus
    average similarity with female attribute words
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


# --------------------------------------------------
# Assign neutral interpretation labels
# --------------------------------------------------

def get_association_label(score, threshold=NEAR_BALANCED_THRESHOLD):
    """
    Convert the numerical score into a neutral category.
    """

    if score > threshold:
        return "Male-associated"

    elif score < -threshold:
        return "Female-associated"

    else:
        return "Near-balanced"


# --------------------------------------------------
# Analyze one word
# --------------------------------------------------

def analyze_word(word):
    """
    Analyze one word and return its results.
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
        "gender_association_score": float(
            gender_association_score
        ),
        "association_label": association_label
    }


# --------------------------------------------------
# Main analysis
# --------------------------------------------------

if __name__ == "__main__":

    # Load datasets
    occupations = pd.read_csv(OCCUPATIONS_PATH)
    reference_words = pd.read_csv(REFERENCE_PATH)
    exploratory_terms = pd.read_csv(EXPLORATORY_PATH)

    # Remove accidental whitespace from CSV values
    occupations["occupation"] = (
        occupations["occupation"]
        .astype(str)
        .str.strip()
    )

    reference_words["reference_word"] = (
        reference_words["reference_word"]
        .astype(str)
        .str.strip()
    )

    exploratory_terms["term"] = (
        exploratory_terms["term"]
        .astype(str)
        .str.strip()
    )

    # ----------------------------------------------
    # Occupation Analysis
    # ----------------------------------------------

    print("\nGenderLens Occupation Analysis")
    print("------------------------------")

    for word in occupations["occupation"]:

        result = analyze_word(word)

        print(
            f"{result['word']}: "
            f"{result['gender_association_score']:.4f} "
            f"({result['association_label']})"
        )

    # ----------------------------------------------
    # Reference Word Analysis
    # ----------------------------------------------

    print("\nGenderLens Reference Word Analysis")
    print("----------------------------------")

    for word in reference_words["reference_word"]:

        result = analyze_word(word)

        print(
            f"{result['word']}: "
            f"{result['gender_association_score']:.4f} "
            f"({result['association_label']})"
        )

    # ----------------------------------------------
    # Exploratory Term Analysis
    # ----------------------------------------------

    print("\nGenderLens Exploratory Term Analysis")
    print("------------------------------------")

    for word in exploratory_terms["term"]:

        result = analyze_word(word)

        print(
            f"{result['word']}: "
            f"{result['gender_association_score']:.4f} "
            f"({result['association_label']})"
        )
