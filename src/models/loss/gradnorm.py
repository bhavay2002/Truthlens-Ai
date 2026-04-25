from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn as nn

from .base_balancer import BaseBalancer

logger = logging.getLogger(__name__)


class GradNormBalancer(BaseBalancer):

    def __init__(
        self,
        task_names: List[str],
        alpha: float = 1.5,
    ) -> None:
        super().__init__()

        if not task_names:
            raise ValueError("task_names must be non-empty")

        self.task_names = task_names
        self.alpha = float(alpha)

        self.log_weights = nn.Parameter(torch.zeros(len(task_names)))

        self.initial_losses: Dict[str, float] = {}
        self._initialized = False

        self._last_grad_norms: Dict[str, torch.Tensor] = {}

    # =========================================================
    # COMBINE (FIXED)
    # =========================================================

    def combine(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        weights = torch.exp(self.log_weights)

        # ✅ normalize weights (CRITICAL)
        weights = weights * (len(weights) / weights.sum().detach())

        losses = torch.stack([task_losses[t] for t in self.task_names])

        if not self._initialized:
            self.initial_losses = {
                t: float(task_losses[t].detach().item())
                for t in self.task_names
            }
            self._initialized = True

        weighted_losses = weights * losses

        return weighted_losses.sum()

    # =========================================================
    # GRADNORM CORE
    # =========================================================

    def on_before_backward(
        self,
        task_losses: Dict[str, torch.Tensor],
        shared_parameters,
    ) -> None:

        if not self._initialized:
            return

        if shared_parameters is None:
            raise RuntimeError("GradNorm requires shared_parameters")

        weights = torch.exp(self.log_weights)
        weights = weights * (len(weights) / weights.sum().detach())

        grad_norms = []

        for i, task in enumerate(self.task_names):

            loss = task_losses[task]

            grads = torch.autograd.grad(
                weights[i] * loss,
                shared_parameters,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )

            valid_grads = [g.norm() for g in grads if g is not None]

            if not valid_grads:
                grad_norm = torch.tensor(0.0, device=weights.device)
            else:
                grad_norm = torch.stack(valid_grads).mean()

            grad_norms.append(grad_norm)

        grad_norms = torch.stack(grad_norms)

        self._last_grad_norms = {
            t: g.detach() for t, g in zip(self.task_names, grad_norms)
        }

        current_losses = torch.tensor(
            [task_losses[t].detach().item() for t in self.task_names],
            device=grad_norms.device,
        )

        initial_losses = torch.tensor(
            [self.initial_losses[t] for t in self.task_names],
            device=grad_norms.device,
        )

        loss_ratios = current_losses / initial_losses.clamp_min(1e-8)

        avg_ratio = loss_ratios.mean()
        relative_rates = loss_ratios / avg_ratio

        avg_grad_norm = grad_norms.mean().detach()

        target_grad_norms = avg_grad_norm * (relative_rates ** self.alpha)

        grad_loss = torch.nn.functional.l1_loss(
            grad_norms,
            target_grad_norms.detach(),
        )

        grad_loss.backward(retain_graph=True)

    # =========================================================
    # DEBUG
    # =========================================================

    def get_last_grad_norms(self) -> Dict[str, float]:
        return {
            k: float(v.item()) for k, v in self._last_grad_norms.items()
        }

    def get_weights(self) -> Dict[str, float]:
        weights = torch.exp(self.log_weights).detach()
        weights = weights * (len(weights) / weights.sum())
        return {
            t: float(w.item())
            for t, w in zip(self.task_names, weights)
        }