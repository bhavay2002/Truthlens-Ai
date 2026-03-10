"""
Data augmentation for NLP datasets
Improves generalization for fake news detection
"""

import random
import nltk
from nltk.corpus import wordnet

nltk.download("wordnet", quiet=True)

# ------------------------------------------------
# Synonym Replacement
# ------------------------------------------------

def synonym_replacement(text, n=2):

    words = text.split()
    new_words = words.copy()

    random_words = list(set(words))
    random.shuffle(random_words)

    replaced = 0

    for word in random_words:

        synonyms = wordnet.synsets(word)

        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()

            if synonym != word:
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

    if len(words) == 1:
        return text

    new_words = [
        word for word in words
        if random.random() > p
    ]

    if len(new_words) == 0:
        return random.choice(words)

    return " ".join(new_words)


# ------------------------------------------------
# Augment Dataset
# ------------------------------------------------

def augment_dataset(df, text_column="text", multiplier=1):

    augmented_rows = []

    for _, row in df.iterrows():

        text = row[text_column]

        for _ in range(multiplier):

            aug_text = synonym_replacement(text)
            new_row = row.copy()
            new_row[text_column] = aug_text

            augmented_rows.append(new_row)

    augmented_df = df.append(augmented_rows, ignore_index=True)

    return augmented_df