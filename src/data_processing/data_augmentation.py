"""
TruthLens task-aware data augmentation.

Notes:
- No module-level ``random.seed`` (it polluted the global RNG).
- No module-level ``nltk.download`` (silent network call at import).
  Resources are downloaded lazily on first call.
- Uses a per-call ``random.Random`` instance for reproducibility without
  global side-effects.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, List, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# =========================================================
# CONFIG
# =========================================================

@dataclass
class AugmentationConfig:
    multiplier: float = 1.5
    enable_heavy_ops: bool = False  # MLM + embeddings
    similarity_threshold: float = 0.75
    random_seed: int = 42


# =========================================================
# LAZY RESOURCES
# =========================================================

_STOPWORDS: Optional[set] = None
_MLM = None
_EMBEDDER = None
_NLTK_READY = False


def _ensure_nltk():
    global _STOPWORDS, _NLTK_READY
    if _NLTK_READY:
        return
    try:
        import nltk
        from nltk.corpus import stopwords

        for pkg in ("wordnet", "stopwords", "omw-1.4"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        _STOPWORDS = set(stopwords.words("english"))
    except Exception:
        _STOPWORDS = set()
    _NLTK_READY = True


def _get_synonyms(word: str) -> List[str]:
    _ensure_nltk()
    try:
        from nltk.corpus import wordnet
    except Exception:
        return []
    syns = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            w = lemma.name().replace("_", " ")
            if w != word:
                syns.add(w)
    return list(syns)


def _get_mlm():
    global _MLM
    if _MLM is None:
        try:
            from transformers import pipeline
            _MLM = pipeline("fill-mask", model="roberta-base", top_k=3)
        except Exception:
            _MLM = False  # mark as failed
    return _MLM if _MLM is not False else None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMBEDDER = False
    return _EMBEDDER if _EMBEDDER is not False else None


# =========================================================
# BASIC OPS  (each receives a Random instance)
# =========================================================

def synonym_replacement(text: str, rng: random.Random) -> str:
    _ensure_nltk()
    stop = _STOPWORDS or set()
    words = text.split()
    indices = list(range(len(words)))
    rng.shuffle(indices)
    for i in indices:
        w = words[i]
        if w.lower() not in stop and len(w) > 3:
            syns = _get_synonyms(w)
            if syns:
                words[i] = rng.choice(syns)
                break
    return " ".join(words)


def random_deletion(text: str, rng: random.Random, p: float = 0.1) -> str:
    words = text.split()
    if len(words) < 5:
        return text
    return " ".join(w for w in words if rng.random() > p)


def random_swap(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 3:
        return text
    i, j = rng.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return " ".join(words)


def ideology_frame_shift(text: str, rng: random.Random) -> str:
    return f"In a broader ideological context, {text}"


def propaganda_injection(text: str, rng: random.Random) -> str:
    return f"Clearly, {text}"


def narrative_reframe(text: str, rng: random.Random) -> str:
    return f"From another perspective, {text}"


def emotion_amplify(text: str, rng: random.Random) -> str:
    return f"{text} This is extremely emotional."


def bias_injection(text: str, rng: random.Random) -> str:
    return f"{text} Obviously biased."


# =========================================================
# HEAVY OPS (OPTIONAL)
# =========================================================

def contextual_replacement(text: str, rng: random.Random) -> str:
    mlm = _get_mlm()
    if mlm is None:
        return text
    words = text.split()
    if len(words) < 6:
        return text
    idx = rng.randint(0, len(words) - 1)
    words[idx] = "<mask>"
    try:
        preds = mlm(" ".join(words))
        words[idx] = preds[0]["token_str"]
    except Exception:
        return text
    return " ".join(words)


def semantic_valid(original: str, augmented: str, threshold: float) -> bool:
    embedder = _get_embedder()
    if embedder is None:
        return True  # cannot check; accept
    from sklearn.metrics.pairwise import cosine_similarity
    emb = embedder.encode([original, augmented])
    score = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return score >= threshold


# =========================================================
# TASK ROUTING
# =========================================================

TASK_OPS: Dict[str, List[Callable]] = {
    "bias": [synonym_replacement, random_deletion],
    "ideology": [ideology_frame_shift],
    "propaganda": [propaganda_injection],
    "frame": [random_swap],
    "narrative": [narrative_reframe],
    "emotion": [emotion_amplify],
}


def select_operation(task: str, config: AugmentationConfig, rng: random.Random):
    ops = list(TASK_OPS.get(task, []))
    if config.enable_heavy_ops:
        ops.append(contextual_replacement)
    if not ops:
        raise ValueError(f"No augmentation ops for task: {task}")
    return rng.choice(ops)


# =========================================================
# CORE
# =========================================================

def augment_text(
    text: str,
    *,
    task: str,
    config: AugmentationConfig,
    rng: random.Random,
) -> str:
    text = str(text).strip()
    if not text:
        return text
    op = select_operation(task, config, rng)
    augmented = op(text, rng)
    if config.enable_heavy_ops:
        if not semantic_valid(text, augmented, config.similarity_threshold):
            return text
    return augmented


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

    rng = random.Random(config.random_seed)
    records = df.to_dict("records")
    extra = int(len(records) * (config.multiplier - 1))

    augmented: List[Dict] = []
    for _ in range(extra):
        row = rng.choice(records).copy()
        row[text_column] = augment_text(
            row[text_column], task=task, config=config, rng=rng,
        )
        augmented.append(row)

    result = pd.concat([df, pd.DataFrame(augmented)], ignore_index=True)

    logger.info(
        "Augmented | task=%s | original=%d | added=%d | total=%d",
        task, len(df), len(augmented), len(result),
    )
    return result
