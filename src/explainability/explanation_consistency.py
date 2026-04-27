from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.explainability.utils_validation import validate_tokens_scores

logger = logging.getLogger(__name__)
EPS = 1e-12


class ExplanationConsistency:

    # =====================================================
    # MAP CONVERSION
    # =====================================================

    @staticmethod
    def _as_map_with_conf(
        items: Optional[List[Dict]],
        key: str,
    ) -> Tuple[Optional[Dict[str, float]], float]:

        if not items:
            return None, 0.0

        def _get(item, attr, default=None):
            if isinstance(item, dict):
                return item.get(attr, default)
            return getattr(item, attr, default)

        m = {str(_get(i, "token")): float(_get(i, key, 0.0) or 0.0) for i in items}

        # extract confidence if available (optional)
        conf = float(_get(items[0], "confidence", 1.0) or 1.0)

        return m, np.clip(conf, 0.0, 1.0)

    @staticmethod
    def _lime_map_with_conf(
        items: Optional[List],
    ) -> Tuple[Optional[Dict[str, float]], float]:

        if not items:
            return None, 0.0

        m = {str(t): float(s) for t, s in items}
        return m, 1.0  # LIME raw has no confidence

    # =====================================================
    # NORMALIZATION
    # =====================================================

    @staticmethod
    def _normalize(v: Dict[str, float]) -> Dict[str, float]:
        vals = np.array(list(v.values()), dtype=float)
        vals = np.abs(vals)
        vals = vals / (np.sum(vals) + EPS)
        return dict(zip(v.keys(), vals))

    # =====================================================
    # METRICS
    # =====================================================

    @staticmethod
    def _pearson(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    @staticmethod
    def _spearman(a, b):
        return float(np.corrcoef(np.argsort(a), np.argsort(b))[0, 1])

    @staticmethod
    def _cosine(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + EPS
        return float(np.dot(a, b) / denom)

    # =====================================================
    # CORE
    # =====================================================

    def _compare(
        self,
        a: Dict[str, float],
        b: Dict[str, float],
        conf_a: float,
        conf_b: float,
    ) -> Dict[str, float]:

        common = sorted(set(a.keys()) & set(b.keys()))

        if len(common) < 2:
            return {"pearson": 0.0, "spearman": 0.0, "cosine": 0.0}

        va = np.array([a[t] for t in common], dtype=float)
        vb = np.array([b[t] for t in common], dtype=float)

        validate_tokens_scores(common, va.tolist())

        # raw correlations
        p = self._pearson(va, vb)
        s = self._spearman(va, vb)
        c = self._cosine(va, vb)

        # 🔥 confidence weighting
        w = min(conf_a, conf_b)

        return {
            "pearson": float(p * w),
            "spearman": float(s * w),
            "cosine": float(c * w),
        }

    # =====================================================
    # TOKEN-LEVEL CONSISTENCY
    # =====================================================

    def _token_consistency(self, sources: Dict[str, Dict[str, float]]):

        tokens = sorted(set().union(*[set(s.keys()) for s in sources.values()]))

        token_scores = {}

        for t in tokens:

            vals = [src[t] for src in sources.values() if t in src]

            if len(vals) < 2:
                token_scores[t] = 0.0
                continue

            vals = np.array(vals)

            score = 1.0 - float(np.std(vals))
            token_scores[t] = float(np.clip(score, 0.0, 1.0))

        return token_scores

    # =====================================================
    # MAIN
    # =====================================================

    def compute(
        self,
        shap_importance: Optional[List[Dict]] = None,
        integrated_gradients: Optional[List[Dict]] = None,
        attention_scores: Optional[List[Dict]] = None,
        lime_importance: Optional[List] = None,
    ) -> Dict[str, float]:

        shap_m, shap_c = self._as_map_with_conf(shap_importance, "importance")
        ig_m, ig_c = self._as_map_with_conf(integrated_gradients, "importance")
        att_m, att_c = self._as_map_with_conf(attention_scores, "importance")
        lime_m, lime_c = self._lime_map_with_conf(lime_importance)

        sources = {}
        confidences = {}

        if shap_m:
            sources["shap"] = self._normalize(shap_m)
            confidences["shap"] = shap_c

        if ig_m:
            sources["ig"] = self._normalize(ig_m)
            confidences["ig"] = ig_c

        if att_m:
            sources["att"] = self._normalize(att_m)
            confidences["att"] = att_c

        if lime_m:
            sources["lime"] = self._normalize(lime_m)
            confidences["lime"] = lime_c

        if len(sources) < 2:
            return {}

        # -------------------------
        # pairwise
        # -------------------------
        results = {}
        keys = list(sources.keys())

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]

                metrics = self._compare(
                    sources[k1],
                    sources[k2],
                    confidences[k1],
                    confidences[k2],
                )

                for m, v in metrics.items():
                    results[f"{k1}_vs_{k2}_{m}"] = v

        # -------------------------
        # overall agreement
        # -------------------------
        overall = float(np.mean(list(results.values()))) if results else 0.0

        # -------------------------
        # token-level
        # -------------------------
        token_scores = self._token_consistency(sources)

        results["overall_agreement"] = overall
        results["token_agreement_mean"] = float(np.mean(list(token_scores.values())))

        return results