import pandas as pd
from gensim.models import KeyedVectors


# Path to the pretrained Word2Vec model
MODEL_PATH = "models/GoogleNews-vectors-negative300.bin"

# Path to the occupation dataset
DATA_PATH = "data/occupations.csv"


# Load the pretrained Word2Vec model
print("Loading Word2Vec model...")

model = KeyedVectors.load_word2vec_format(
    MODEL_PATH,
    binary=True
)

print("Model loaded successfully.")


# Load occupation dataset
occupations = pd.read_csv(DATA_PATH)


# Check which occupations exist in the model vocabulary
available_words = []
missing_words = []

for word in occupations["occupation"]:

    if word in model.key_to_index:
        available_words.append(word)
    else:
        missing_words.append(word)


# Display results
print("\nOccupation vocabulary check")
print("---------------------------")

print("\nAvailable words:")
for word in available_words:
    print(f"✓ {word}")

print("\nMissing words:")
for word in missing_words:
    print(f"✗ {word}")


# Summary
print("\nSummary")
print("-------")
print("Total occupations:", len(occupations))
print("Available:", len(available_words))
print("Missing:", len(missing_words))
