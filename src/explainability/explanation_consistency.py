from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.explainability.utils_validation import validate_tokens_scores

logger = logging.getLogger(__name__)


class ExplanationConsistency:
    @staticmethod
    def _as_map(items: Optional[List[Dict]], key: str) -> Optional[Dict[str, float]]:
        if not items:
            return None
        return {str(i.get("token")): float(i.get(key, 0.0)) for i in items}

    @staticmethod
    def _lime_map(items: Optional[List]) -> Optional[Dict[str, float]]:
        if not items:
            return None
        return {str(t): float(s) for t, s in items}

    @staticmethod
    def _corr(a: Dict[str, float], b: Dict[str, float]) -> float:
        common = sorted(set(a.keys()) & set(b.keys()))
        if len(common) < 2:
            return 0.0
        va = np.array([a[t] for t in common], dtype=float)
        vb = np.array([b[t] for t in common], dtype=float)
        validate_tokens_scores(common, va.tolist())
        if np.std(va) < 1e-12 or np.std(vb) < 1e-12:
            return 0.0
        c = np.corrcoef(va, vb)[0, 1]
        return 0.0 if np.isnan(c) else float(c)

    def compute(
        self,
        shap_importance: Optional[List[Dict]] = None,
        integrated_gradients: Optional[List[Dict]] = None,
        attention_scores: Optional[List[Dict]] = None,
        lime_importance: Optional[List] = None,
    ) -> Dict[str, float]:
        shap_m = self._as_map(shap_importance, "importance")
        ig_m = self._as_map(integrated_gradients, "importance")
        att_m = self._as_map(attention_scores, "attention")
        lime_m = self._lime_map(lime_importance)

        out: Dict[str, float] = {}
        if shap_m and ig_m:
            out["shap_vs_ig"] = self._corr(shap_m, ig_m)
        if shap_m and att_m:
            out["shap_vs_attention"] = self._corr(shap_m, att_m)
        if ig_m and lime_m:
            out["ig_vs_lime"] = self._corr(ig_m, lime_m)
        if shap_m and lime_m:
            out["shap_vs_lime"] = self._corr(shap_m, lime_m)
        if ig_m and att_m:
            out["ig_vs_attention"] = self._corr(ig_m, att_m)
        return out
