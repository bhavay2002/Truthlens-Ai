"""
Data augmentation for NLP datasets
Improves generalization for fake news detection
"""

import random
import pandas as pd
import nltk
from nltk.corpus import wordnet, stopwords
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data with error handling
try:
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}. Using empty stopwords set.")
    STOPWORDS = set()

random.seed(42)


# ------------------------------------------------
# Get Synonyms
# ------------------------------------------------

def get_synonyms(word):

    synonyms = set()

    for syn in wordnet.synsets(word):

        for lemma in syn.lemmas():

            synonym = lemma.name().replace("_", " ").lower()

            if synonym != word:
                synonyms.add(synonym)

    return list(synonyms)


# ------------------------------------------------
# Synonym Replacement
# ------------------------------------------------

def synonym_replacement(text, n=2):

    words = text.split()

    new_words = words.copy()

    candidates = [
        word for word in words
        if word not in STOPWORDS and len(word) > 3
    ]

    random.shuffle(candidates)

    replaced = 0

    for word in candidates:

        synonyms = get_synonyms(word)

        if len(synonyms) > 0:

            synonym = random.choice(synonyms)

            new_words = [
                synonym if w == word else w
                for w in new_words
            ]

            replaced += 1

        if replaced >= n:
            break

    return " ".join(new_words)


# ------------------------------------------------
# Random Deletion
# ------------------------------------------------

def random_deletion(text, p=0.1):

    words = text.split()

    if len(words) <= 5:
        return text

    new_words = [
        word for word in words
        if random.random() > p
    ]

    if len(new_words) == 0:
        return random.choice(words)

    return " ".join(new_words)


# ------------------------------------------------
# Random Swap
# ------------------------------------------------

def random_swap(text):

    words = text.split()

    if len(words) < 3:
        return text

    idx1 = random.randint(0, len(words) - 1)
    idx2 = random.randint(0, len(words) - 1)

    words[idx1], words[idx2] = words[idx2], words[idx1]

    return " ".join(words)


# ------------------------------------------------
# Augment Single Text
# ------------------------------------------------

def augment_text(text):

    operations = [
        synonym_replacement,
        random_deletion,
        random_swap
    ]

    operation = random.choice(operations)

    return operation(text)


# ------------------------------------------------
# Augment Dataset
# ------------------------------------------------

def augment_dataset(df, text_column="text", multiplier=1):

    augmented_rows = []

    for _, row in df.iterrows():

        text = row[text_column]

        for _ in range(multiplier):

            aug_text = augment_text(text)

            new_row = row.copy()

            new_row[text_column] = aug_text

            augmented_rows.append(new_row)

    augmented_df = pd.concat(
        [df, pd.DataFrame(augmented_rows)],
        ignore_index=True
    )

    return augmented_df