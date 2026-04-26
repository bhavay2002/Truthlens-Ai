"""
TruthLens Task-Aware Data Augmentation (Production + Research Ready)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, List, Dict, Optional

import pandas as pd
import nltk

from nltk.corpus import wordnet, stopwords

# Optional heavy deps
try:
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    pipeline = None
    SentenceTransformer = None

logger = logging.getLogger(__name__)

# =========================================================
# CONFIG
# =========================================================

@dataclass
class AugmentationConfig:
    multiplier: float = 1.5
    enable_heavy_ops: bool = False   # MLM + embeddings
    similarity_threshold: float = 0.75
    random_seed: int = 42


# =========================================================
# INIT
# =========================================================

random.seed(42)

try:
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()

_mlm = None
_embedder = None


def get_mlm():
    global _mlm
    if _mlm is None and pipeline:
        _mlm = pipeline("fill-mask", model="roberta-base", top_k=3)
    return _mlm


def get_embedder():
    global _embedder
    if _embedder is None and SentenceTransformer:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# =========================================================
# BASIC OPS
# =========================================================

def get_synonyms(word: str):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            w = lemma.name().replace("_", " ")
            if w != word:
                synonyms.add(w)
    return list(synonyms)


def synonym_replacement(text: str) -> str:
    words = text.split()
    for i, w in enumerate(words):
        if w.lower() not in STOPWORDS and len(w) > 3:
            syns = get_synonyms(w)
            if syns:
                words[i] = random.choice(syns)
                break
    return " ".join(words)


def random_deletion(text: str, p=0.1):
    words = text.split()
    if len(words) < 5:
        return text
    return " ".join([w for w in words if random.random() > p])


def random_swap(text: str):
    words = text.split()
    if len(words) < 3:
        return text
    i, j = random.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return " ".join(words)


# =========================================================
# TASK-SPECIFIC OPS
# =========================================================

def ideology_frame_shift(text: str) -> str:
    return f"In a broader ideological context, {text}"


def propaganda_injection(text: str) -> str:
    return f"Clearly, {text}"


def narrative_reframe(text: str) -> str:
    return f"From another perspective, {text}"


def emotion_amplify(text: str) -> str:
    return f"{text} This is extremely emotional."


def bias_injection(text: str) -> str:
    return f"{text} Obviously biased."


# =========================================================
# HEAVY OPS (OPTIONAL)
# =========================================================

def contextual_replacement(text: str) -> str:
    mlm = get_mlm()
    if mlm is None:
        return text

    words = text.split()
    if len(words) < 6:
        return text

    idx = random.randint(0, len(words) - 1)
    words[idx] = "<mask>"

    try:
        preds = mlm(" ".join(words))
        words[idx] = preds[0]["token_str"]
    except Exception:
        return text

    return " ".join(words)


def semantic_valid(original: str, augmented: str, threshold: float) -> bool:
    embedder = get_embedder()
    if embedder is None:
        return True  # skip check

    emb = embedder.encode([original, augmented])
    score = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return score >= threshold


# =========================================================
# TASK-AWARE ROUTING
# =========================================================

TASK_OPS: Dict[str, List[Callable[[str], str]]] = {
    "bias": [synonym_replacement, random_deletion],
    "ideology": [ideology_frame_shift],
    "propaganda": [propaganda_injection],
    "frame": [random_swap],
    "narrative": [narrative_reframe],
    "emotion": [emotion_amplify],
}


def select_operation(task: str, config: AugmentationConfig):
    ops = TASK_OPS.get(task, [])

    if config.enable_heavy_ops:
        ops = ops + [contextual_replacement]

    if not ops:
        raise ValueError(f"No ops for task: {task}")

    return random.choice(ops)


# =========================================================
# CORE
# =========================================================

def augment_text(
    text: str,
    *,
    task: str,
    config: AugmentationConfig,
) -> str:

    text = str(text).strip()
    if not text:
        return text

    op = select_operation(task, config)

    augmented = op(text)

    if config.enable_heavy_ops:
        if not semantic_valid(text, augmented, config.similarity_threshold):
            return text

    return augmented


# =========================================================
# DATASET
# =========================================================

def augment_dataset(
    df: pd.DataFrame,
    *,
    task: str,
    text_column: str = "text",
    config: Optional[AugmentationConfig] = None,
) -> pd.DataFrame:

    config = config or AugmentationConfig()

    if config.multiplier <= 1:
        return df.copy()

    records = df.to_dict("records")
    extra = int(len(records) * (config.multiplier - 1))

    augmented = []

    for _ in range(extra):

        row = random.choice(records)
        new_row = row.copy()

        new_row[text_column] = augment_text(
            row[text_column],
            task=task,
            config=config,
        )

        augmented.append(new_row)

    result = pd.concat(
        [df, pd.DataFrame(augmented)],
        ignore_index=True,
    )

    logger.info(
        "Augmented | task=%s | original=%d | added=%d | total=%d",
        task,
        len(df),
        len(augmented),
        len(result),
    )

    return result