# src/features/syntactic_features.py

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.numerics import normalized_entropy
from src.features.base.spacy_loader import get_shared_nlp
from src.features.base.tokenization import ensure_tokens_word

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _simple_sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _memoized_dependency_depths(tokens) -> List[int]:
    """Return the dep-tree depth of every token in O(N) amortised time.

    The previous implementation walked from each token up to the root
    independently, paying O(N) per token and O(N^2) per document. For
    documents on the order of a few thousand tokens (common for long
    articles processed by the misinformation pipeline) this dominated the
    syntactic extractor's wall-clock cost.

    Here we cache each token's depth keyed by ``token.i`` and re-use
    cached ancestors when ascending. Each token is therefore visited at
    most twice across the whole pass.
    """
    depth_cache: Dict[int, int] = {}
    out: List[int] = []

    for token in tokens:
        # Build the ascent chain up until the cache hits or we reach the
        # root (a token whose head is itself, per spaCy's convention).
        chain: List[Any] = []
        cur = token
        seen = set()
        # Cap defensively: pathological deps + long sentences should not
        # turn this into an O(depth^2) re-walk.
        while cur.i not in depth_cache and cur.head != cur and len(chain) < 100:
            if cur.i in seen:
                break
            seen.add(cur.i)
            chain.append(cur)
            cur = cur.head

        # ``cur`` is now either a cached node or the root.
        if cur.i in depth_cache:
            base = depth_cache[cur.i]
        else:
            # Root or the cycle-break sentinel; treat as depth 0.
            base = 0
            depth_cache[cur.i] = 0

        # Backfill cache top-down so subsequent ascents short-circuit.
        for t in reversed(chain):
            base += 1
            depth_cache[t.i] = base

        out.append(depth_cache[token.i])

    return out


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class SyntacticFeatures(BaseFeature):

    name: str = "syntactic_features"
    group: str = "syntactic"
    description: str = "Advanced syntactic structure features"

    _nlp: Any = field(default=None, init=False, repr=False)
    _spacy_available: bool = field(default=False, init=False, repr=False)

    # -----------------------------------------------------

    def initialize(self) -> None:
        if self._nlp is not None or self._spacy_available:
            return
        self._nlp = get_shared_nlp("en_core_web_sm")
        self._spacy_available = self._nlp is not None

    # -----------------------------------------------------
    # spaCy version
    # -----------------------------------------------------

    def _extract_spacy_doc(self, doc) -> Dict[str, float]:

        # Audit fix §5.2 — both the token list and the per-sentence
        # length tally now filter tokens the same way (drop both is_space
        # and is_punct). Previously the document-level loop dropped only
        # ``is_space`` while the sentence-level loop dropped ``is_punct``
        # too, so ``avg_len * num_sentences`` did not equal ``n``.
        def _is_content_token(t) -> bool:
            return not (t.is_space or t.is_punct)

        tokens = [t for t in doc if _is_content_token(t)]
        n = len(tokens) or 1

        # -------------------------
        # POS DISTRIBUTION
        # -------------------------

        pos_counts = Counter(t.pos_ for t in tokens)

        pos_keys = ["NOUN", "VERB", "ADJ", "ADV"]
        pos_vals = np.array([pos_counts.get(k, 0) for k in pos_keys], dtype=np.float32)

        pos_probs = pos_vals / (pos_vals.sum() + EPS)

        # entropy
        pos_entropy = normalized_entropy(pos_probs)

        # -------------------------
        # SENTENCE STRUCTURE
        # -------------------------

        sentences = list(doc.sents)

        # Same content-token filter as the document-level pass above so
        # the per-sentence lengths are consistent with ``n``.
        lengths = np.array(
            [sum(1 for t in s if _is_content_token(t)) for s in sentences],
            dtype=np.float32,
        )

        avg_len = float(lengths.mean()) if lengths.size else 0.0
        std_len = float(lengths.std()) if lengths.size else 0.0

        # normalized dispersion
        dispersion = std_len / (avg_len + EPS)

        # entropy of sentence lengths
        if lengths.size > 1:
            probs = lengths / (lengths.sum() + EPS)
            sent_entropy = normalized_entropy(probs)
        else:
            sent_entropy = 0.0

        # -------------------------
        # SYNTACTIC COMPLEXITY
        # -------------------------

        depths = _memoized_dependency_depths(tokens)

        complexity = float(np.mean(depths)) if depths else 0.0

        # -------------------------
        # COORDINATION / SUBORDINATION
        # -------------------------

        conj = sum(1 for t in tokens if t.dep_ == "conj")
        subord = sum(1 for t in tokens if t.dep_ in {"ccomp", "advcl", "relcl"})

        coord_ratio = conj / (n + EPS)
        subord_ratio = subord / (n + EPS)

        # -------------------------
        # OUTPUT
        # -------------------------
        # Audit fix §1.1 — emit RAW magnitudes for length / complexity.
        # Population-level scaling is the FeatureScalingPipeline's job;
        # the per-extractor /50.0 and /10.0 magic divisors that used to
        # live here pre-scaled the value into [0, 1] using a constant
        # picked by hand and therefore drifted as the corpus changed.

        return {
            "syn_pos_entropy": self._safe(pos_entropy),

            "syn_sentence_avg_len": self._safe_unbounded(avg_len),
            "syn_sentence_dispersion": self._safe(dispersion),
            "syn_sentence_entropy": self._safe(sent_entropy),

            "syn_complexity": self._safe_unbounded(complexity),

            "syn_coordination": self._safe(coord_ratio),
            "syn_subordination": self._safe(subord_ratio),

            # Audit fix §3.6 — emit an explicit availability indicator
            # so the downstream model can attenuate syntactic signal on
            # the rows where spaCy was unavailable instead of having to
            # learn the "spaCy-was-up" pattern from the bimodal cliff
            # in the other syn_* columns.
            "syn_spacy_available": 1.0,
        }

    # -----------------------------------------------------
    # fallback
    # -----------------------------------------------------

    def _extract_fallback(self, context: FeatureContext) -> Dict[str, float]:
        # Audit fix §3.6 — when spaCy is unavailable the POS-tag /
        # dep-tree / sentence-level features can't be computed at all.
        # The previous fallback hard-coded them to 0.0, which produced a
        # bimodal distribution (real values vs constant 0) that the
        # model learned as a spurious "spaCy-was-up" signal. We now
        # emit only the spaCy-free columns (`syn_sentence_avg_len`
        # using the regex token + simple sentence split) plus the
        # `syn_spacy_available=0.0` indicator. The dropped keys are
        # imputed downstream by `FeatureSchemaValidator` (fill_value /
        # training-set mean via `FeatureScalingPipeline`).
        text = context.text
        tokens = ensure_tokens_word(context)
        sentences = _simple_sentence_split(text)

        n = len(tokens) or 1

        avg_len = n / len(sentences) if sentences else n

        return {
            "syn_sentence_avg_len": self._safe_unbounded(float(avg_len)),
            "syn_spacy_available": 0.0,
        }

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        self.initialize()

        if self._spacy_available and self._nlp is not None:
            doc = self._nlp(text)
            return self._extract_spacy_doc(doc)

        return self._extract_fallback(context)

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))

    # -----------------------------------------------------

    def _safe_unbounded(self, v: float) -> float:
        """Return a finite, non-negative value with no upper clip.

        Audit fix §1.1 — magnitudes such as ``avg_sentence_length`` and
        ``dependency_depth`` are emitted raw so the
        :class:`FeatureScalingPipeline` can fit a corpus-aware
        normalisation. We still drop NaN / inf and floor at zero so a
        broken extractor cannot poison downstream scaling.
        """
        if not np.isfinite(v) or v < 0:
            return 0.0
        return float(v)