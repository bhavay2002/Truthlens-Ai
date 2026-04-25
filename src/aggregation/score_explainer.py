from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# SECTION KEYWORDS
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
# UTILS
# =========================================================

def _normalize_importance(scores: np.ndarray) -> np.ndarray:
    scores = scores / (np.sum(np.abs(scores)) + EPS)
    return scores


def _entropy(probs):
    probs = np.asarray(probs)
    return -np.sum(probs * np.log(probs + EPS))


# =========================================================
# EXPLAINER
# =========================================================

class ScoreExplainer:

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Any = None,
        *,
        device: Optional[str] = None,
        steps: int = 32,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = steps

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    # =====================================================
    # FAST INTEGRATED GRADIENTS
    # =====================================================

    def _integrated_gradients(self, input_ids, attention_mask, task, target_idx):

        embeddings = self.model.encoder.embeddings(input_ids.to(self.device))
        baseline = torch.zeros_like(embeddings)

        scaled = [
            baseline + (i / self.steps) * (embeddings - baseline)
            for i in range(self.steps)
        ]

        scaled = torch.cat(scaled, dim=0).requires_grad_(True)

        attention_mask = attention_mask.repeat(self.steps, 1)

        outputs = self.model.encoder(
            inputs_embeds=scaled,
            attention_mask=attention_mask.to(self.device),
        )

        cls = outputs.last_hidden_state[:, 0]
        logits = self.model.heads[task](cls)

        target = logits[:, target_idx].sum()

        self.model.zero_grad()
        target.backward()

        grads = scaled.grad.view(self.steps, *embeddings.shape).mean(dim=0)

        integrated = (embeddings - baseline) * grads

        importance = integrated.sum(dim=-1).detach().cpu().numpy().squeeze()

        return _normalize_importance(importance)

    # =====================================================
    # TOKEN → SECTION AGGREGATION
    # =====================================================

    def _section_scores(self, tokens, importance):

        section_scores = {k: 0.0 for k in SECTION_KEYWORDS}

        for token, score in zip(tokens, importance):

            clean = token.lower().replace("##", "")

            for section, keys in SECTION_KEYWORDS.items():
                if any(k in clean for k in keys):
                    section_scores[section] += float(score)

        return section_scores

    # =====================================================
    # MAIN
    # =====================================================

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

        results = {}

        for task, output in predictor_output.items():

            logits = output.get("logits")
            probs = output.get("probabilities")

            if logits is None:
                continue

            logits = torch.tensor(logits)
            target_idx = int(torch.argmax(logits))

            importance = self._integrated_gradients(
                input_ids,
                attention_mask,
                task,
                target_idx,
            )

            section_scores = self._section_scores(tokens, importance)

            # uncertainty
            uncertainty = None
            if probs is not None:
                uncertainty = float(_entropy(probs))

            results[task] = {
                "top_tokens": sorted(
                    zip(tokens, importance),
                    key=lambda x: -abs(x[1])
                )[:top_k],
                "section_scores": section_scores,
                "uncertainty": uncertainty,
            }

        return results

    # =====================================================
    # PROFILE MODE (ENHANCED)
    # =====================================================

    def explain_profile(
        self,
        profile: Dict[str, Any],
        *,
        top_k: int = 5,
    ) -> Dict[str, Any]:

        contributions = []

        for section, payload in profile.items():
            if isinstance(payload, dict):
                for k, v in payload.items():
                    try:
                        val = float(v)
                        contributions.append((section, k, val))
                    except:
                        continue

        contributions.sort(key=lambda x: -abs(x[2]))

        return {
            "top_features": contributions[:top_k],
            "section_scores": {
                s: sum(v for sec, _, v in contributions if sec == s)
                for s in set(sec for sec, _, _ in contributions)
            },
        }