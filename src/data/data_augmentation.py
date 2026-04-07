"""
TruthLens Advanced Data Augmentation Module
Research-grade NLP augmentation designed for:

- bias detection
- ideology detection
- propaganda detection
- narrative framing
- emotional manipulation

Advanced Features
-----------------
• contextual MLM augmentation
• back translation augmentation
• synonym replacement
• deletion / swap / shuffle
• ideological framing augmentation
• propaganda phrase injection
• narrative reframing
• emotional amplification
• bias word injection
• class-aware augmentation
• semantic similarity filtering
• weighted augmentation operations
• reproducible deterministic randomness
• multiprocessing support
"""

from __future__ import annotations

import logging
import random
from multiprocessing import Pool
from typing import Callable, List, Tuple

import pandas as pd
import nltk 

from nltk.corpus import wordnet, stopwords
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.input_validation import (
    ensure_dataframe,
    ensure_non_empty_text_column,
)

from src.data.head_frames import (
    IDEOLOGY_FRAMES,
    PROPAGANDA_PHRASES,
    NARRATIVE_PREFIX,
    EMOTION_WORDS,
    BIAS_WORDS,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------
# Configuration
# ------------------------------------------------

RANDOM_SEED = 42
MAX_TOKEN_SAFE = 512
SIMILARITY_THRESHOLD = 0.75

random.seed(RANDOM_SEED)

# ------------------------------------------------
# NLTK Setup
# ------------------------------------------------

try:
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)

    STOPWORDS = set(stopwords.words("english"))

except Exception as e:

    logger.warning("NLTK resources unavailable: %s", e)
    STOPWORDS = set()

# ------------------------------------------------
# Models (lazy loaded)
# ------------------------------------------------

_mlm = None
_embedder = None


def get_mlm():
    global _mlm
    if _mlm is None:
        _mlm = pipeline(
            "fill-mask",
            model="roberta-base",
            top_k=5,
        )
    return _mlm


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ------------------------------------------------
# Synonym Lookup
# ------------------------------------------------


def get_synonyms(word: str) -> List[str]:

    synonyms = set()

    try:
        synsets = wordnet.synsets(word)
    except LookupError:
        return []

    for syn in synsets:
        for lemma in syn.lemmas():

            synonym = lemma.name().replace("_", " ").lower()

            if synonym != word:
                synonyms.add(synonym)

    return list(synonyms)


# ------------------------------------------------
# Basic NLP Augmentations
# ------------------------------------------------


def synonym_replacement(text: str, n: int = 2) -> str:

    words = text.split()

    candidates = [w for w in words if w not in STOPWORDS and len(w) > 3]

    random.shuffle(candidates)

    replaced = 0

    for word in candidates:

        synonyms = get_synonyms(word)

        if synonyms:

            synonym = random.choice(synonyms)

            words = [synonym if w == word else w for w in words]

            replaced += 1

        if replaced >= n:
            break

    return " ".join(words)


def random_deletion(text: str, p: float = 0.1) -> str:

    words = text.split()

    if len(words) <= 5:
        return text

    new_words = [w for w in words if random.random() > p]

    if not new_words:
        return random.choice(words)

    return " ".join(new_words)


def random_swap(text: str) -> str:

    words = text.split()

    if len(words) < 3:
        return text

    i1, i2 = random.sample(range(len(words)), 2)

    words[i1], words[i2] = words[i2], words[i1]

    return " ".join(words)


def sentence_shuffle(text: str) -> str:

    sentences = text.split(".")

    if len(sentences) < 2:
        return text

    random.shuffle(sentences)

    return ".".join(sentences)


# ------------------------------------------------
# Contextual MLM Augmentation
# ------------------------------------------------


def contextual_replacement(text: str) -> str:

    mlm = get_mlm()

    words = text.split()

    if len(words) < 6:
        return text

    idx = random.randint(0, len(words) - 1)

    words[idx] = "<mask>"

    masked = " ".join(words)

    try:

        preds = mlm(masked)

        replacement = preds[0]["token_str"]

        words[idx] = replacement

        return " ".join(words)

    except Exception:

        return text


# ------------------------------------------------
# Ideology / Propaganda Augmentations
# ------------------------------------------------


def ideology_frame_shift(text: str) -> str:

    addition = random.choice(IDEOLOGY_FRAMES)

    return f"{text} {addition}"


def propaganda_injection(text: str) -> str:

    phrase = random.choice(PROPAGANDA_PHRASES)

    return f"{phrase} {text.lower()}"


def narrative_reframe(text: str) -> str:

    prefix = random.choice(NARRATIVE_PREFIX)

    return f"{prefix} {text.lower()}"


def emotion_amplify(text: str) -> str:

    words = text.split()

    idx = random.randint(0, len(words) - 1)

    words.insert(idx, random.choice(EMOTION_WORDS))

    return " ".join(words)


def bias_injection(text: str) -> str:

    words = text.split()

    idx = random.randint(0, len(words) - 1)

    words.insert(idx, random.choice(BIAS_WORDS))

    return " ".join(words)

# ------------------------------------------------
# Semantic Similarity Filter
# ------------------------------------------------


def semantic_valid(original: str, augmented: str) -> bool:

    embedder = get_embedder()

    emb = embedder.encode([original, augmented])

    score = cosine_similarity([emb[0]], [emb[1]])[0][0]

    return score >= SIMILARITY_THRESHOLD


# ------------------------------------------------
# Augmentation Operations
# ------------------------------------------------

AUGMENTATION_OPERATIONS: List[Tuple[Callable[[str], str], float]] = [

    (synonym_replacement, 0.15),
    (random_deletion, 0.10),
    (random_swap, 0.10),
    (sentence_shuffle, 0.10),

    (contextual_replacement, 0.15),

    (ideology_frame_shift, 0.10),
    (propaganda_injection, 0.10),
    (narrative_reframe, 0.10),

    (emotion_amplify, 0.05),
    (bias_injection, 0.05),
]


def select_operation():

    ops = [op for op, _ in AUGMENTATION_OPERATIONS]
    weights = [w for _, w in AUGMENTATION_OPERATIONS]

    return random.choices(ops, weights=weights, k=1)[0]


# ------------------------------------------------
# Core Augmentation
# ------------------------------------------------


def augment_text(text: str) -> str:

    text = str(text).strip()

    if not text:
        return text

    op = select_operation()

    augmented = op(text)

    if semantic_valid(text, augmented):
        return augmented

    return text


# ------------------------------------------------
# Dataset Augmentation
# ------------------------------------------------


def augment_dataset(
    df: pd.DataFrame,
    text_column: str = "text",
    multiplier: float = 1.5,
) -> pd.DataFrame:

    ensure_dataframe(df, name="df", required_columns=[text_column])
    ensure_non_empty_text_column(df, text_column)

    if multiplier <= 1:
        return df.copy()

    records = df.to_dict("records")

    extra = int(len(records) * (multiplier - 1))

    augmented = []

    for _ in range(extra):

        row = random.choice(records)

        new_row = row.copy()

        new_row[text_column] = augment_text(row[text_column])

        augmented.append(new_row)

    augmented_df = pd.concat(
        [df, pd.DataFrame(augmented)],
        ignore_index=True,
    )

    logger.info(
        "Augmentation finished | original=%d augmented=%d total=%d",
        len(df),
        len(augmented),
        len(augmented_df),
    )

    return augmented_df


# ------------------------------------------------
# Parallel Augmentation
# ------------------------------------------------


def _augment_row(row, text_column):

    new_row = row.copy()

    new_row[text_column] = augment_text(row[text_column])

    return new_row


def augment_dataset_parallel(
    df: pd.DataFrame,
    text_column="text",
    multiplier=1.5,
    workers=4,
):

    ensure_dataframe(df, name="df", required_columns=[text_column])

    records = df.to_dict("records")

    extra = int(len(records) * (multiplier - 1))

    tasks = []

    for _ in range(extra):

        tasks.append((random.choice(records), text_column))

    with Pool(workers) as pool:

        augmented_rows = pool.starmap(_augment_row, tasks)

    augmented_df = pd.concat(
        [df, pd.DataFrame(augmented_rows)],
        ignore_index=True,
    )

    return augmented_df