from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


# =========================================================
# TOKEN → SECTION MAPPING (CRITICAL)
# =========================================================

SECTION_KEYWORDS = {
    "bias": {"bias", "opinion", "subjective"},
    "emotion": {"happy", "sad", "anger", "fear", "joy"},
    "narrative": {"story", "claim", "event"},
    "discourse": {"however", "therefore", "because"},
    "graph": {"relation", "connection"},
    "ideology": {"liberal", "conservative"},
    "analysis": {"evidence", "analysis"},
}


# =========================================================
# EXPLAINER
# =========================================================

class ScoreExplainer:

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Any = None,
        *,
        method: str = "integrated_gradients",
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.method = method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    # =========================================================
    # PROFILE-BASED EXPLANATION (no model required)
    # =========================================================
    def explain_profile(
        self,
        profile: Dict[str, Any],
        *,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Lightweight, profile-only explanation.

        Returns the top contributing sections/features by absolute score.
        Used when no model+tokenizer pair is available (e.g. aggregation-only
        runs).
        """
        contributions: List[Dict[str, Any]] = []

        if isinstance(profile, dict):
            for section, payload in profile.items():
                if isinstance(payload, dict):
                    for feature, value in payload.items():
                        try:
                            score = float(value)
                        except (TypeError, ValueError):
                            continue
                        contributions.append({
                            "section": section,
                            "feature": feature,
                            "importance": score,
                            "direction": "positive" if score >= 0 else "negative",
                        })
                else:
                    try:
                        score = float(payload)
                    except (TypeError, ValueError):
                        continue
                    contributions.append({
                        "section": section,
                        "feature": section,
                        "importance": score,
                        "direction": "positive" if score >= 0 else "negative",
                    })

        contributions.sort(key=lambda c: abs(c["importance"]), reverse=True)
        top = contributions[:top_k]

        return {
            "method": self.method,
            "top_contributors": top,
            "num_features": len(contributions),
        }

    # =========================================================
    # INTEGRATED GRADIENTS
    # =========================================================
    def _integrated_gradients(self, input_ids, attention_mask, task, target_index):
        embeddings = self.model.encoder.embeddings(input_ids.to(self.device))
        baseline = torch.zeros_like(embeddings)

        steps = 50
        grads = []

        for i in range(steps + 1):
            scaled = baseline + (i / steps) * (embeddings - baseline)
            scaled.requires_grad_(True)

            outputs = self.model.encoder(
                inputs_embeds=scaled,
                attention_mask=attention_mask.to(self.device),
            )

            cls = outputs.last_hidden_state[:, 0]
            logits = self.model.heads[task](cls)

            target = logits[:, target_index].sum()

            self.model.zero_grad()
            target.backward()

            grads.append(scaled.grad.detach().cpu().numpy())

        avg_grads = np.mean(grads, axis=0)
        integrated = (embeddings.detach().cpu().numpy() - baseline.cpu().numpy()) * avg_grads

        return np.sum(integrated, axis=-1).squeeze()

    # =========================================================
    # TOKEN → SECTION MAPPING
    # =========================================================
    def _map_tokens_to_sections(
        self,
        tokens: List[str],
        importance: np.ndarray,
    ) -> Dict[str, List[Dict[str, Any]]]:

        section_map: Dict[str, List] = {k: [] for k in SECTION_KEYWORDS}

        for token, score in zip(tokens, importance):

            clean_token = token.lower().replace("##", "")

            for section, keywords in SECTION_KEYWORDS.items():
                if any(k in clean_token for k in keywords):
                    section_map[section].append({
                        "token": token,
                        "importance": float(score),
                        "direction": "positive" if score >= 0 else "negative",
                    })

        return section_map

    # =========================================================
    # MAIN ENTRY (PREDICTOR OUTPUT)
    # =========================================================
    def explain_from_prediction(
        self,
        text: str,
        predictor_output: Dict[str, Any],
        *,
        top_k: int = 5,
    ) -> Dict[str, Any]:

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        explanations = {}

        for task, output in predictor_output.items():

            logits = output.get("logits")
            if logits is None:
                continue

            logits = torch.tensor(logits)
            target_index = int(torch.argmax(logits))

            importance = self._integrated_gradients(
                input_ids,
                attention_mask,
                task,
                target_index,
            )

            section_map = self._map_tokens_to_sections(tokens, importance)

            explanations[task] = {
                "top_tokens": sorted(
                    zip(tokens, importance),
                    key=lambda x: -abs(x[1])
                )[:top_k],
                "sections": section_map,
            }

        return explanations