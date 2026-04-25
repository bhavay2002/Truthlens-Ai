from __future__ import annotations

import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# FGM (Fast Gradient Method)
# =========================================================

class FGM:
    """
    Fast Gradient Method for adversarial training.

    Perturbs embeddings in the direction of the gradient.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 1e-5,
        emb_name: str = "embedding",
    ) -> None:
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup: Dict[str, torch.Tensor] = {}

    def attack(self) -> None:
        """
        Add adversarial perturbation to embeddings.
        """
        for name, param in self.model.named_parameters():

            if param.requires_grad and self.emb_name in name:

                if param.grad is None:
                    continue

                self.backup[name] = param.data.clone()

                grad = param.grad
                norm = torch.norm(grad)

                if norm != 0:
                    r_at = self.epsilon * grad / (norm + EPS)
                    param.data.add_(r_at)

    def restore(self) -> None:
        """
        Restore original embeddings.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        self.backup.clear()


# =========================================================
# PGD (Projected Gradient Descent)
# =========================================================

class PGD:
    """
    Multi-step adversarial training.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 1e-5,
        alpha: float = 1e-6,
        steps: int = 3,
        emb_name: str = "embedding",
    ) -> None:
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.emb_name = emb_name

        self.emb_backup: Dict[str, torch.Tensor] = {}
        self.grad_backup: Dict[str, torch.Tensor] = {}

    def attack(self, is_first_attack: bool = False) -> None:

        for name, param in self.model.named_parameters():

            if param.requires_grad and self.emb_name in name:

                if param.grad is None:
                    continue

                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()

                grad = param.grad
                norm = torch.norm(grad)

                if norm != 0:
                    r_at = self.alpha * grad / (norm + EPS)
                    param.data.add_(r_at)
                    param.data = self._project(name, param.data)

    def restore(self) -> None:

        for name, param in self.model.named_parameters():
            if name in self.emb_backup:
                param.data = self.emb_backup[name]

        self.emb_backup.clear()

    def backup_grad(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.grad_backup:
                param.grad = self.grad_backup[name]

    def _project(self, name: str, param_data: torch.Tensor) -> torch.Tensor:
        """
        Project perturbation to epsilon ball.
        """
        r = param_data - self.emb_backup[name]
        if torch.norm(r) > self.epsilon:
            r = self.epsilon * r / (torch.norm(r) + EPS)
        return self.emb_backup[name] + r


# =========================================================
# FREE ADVERSARIAL TRAINING (FAST)
# =========================================================

class FreeAT:
    """
    Free Adversarial Training (reuses gradient).
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 1e-5,
        emb_name: str = "embedding",
    ) -> None:
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.delta: Dict[str, torch.Tensor] = {}

    def attack(self) -> None:

        for name, param in self.model.named_parameters():

            if param.requires_grad and self.emb_name in name:

                if param.grad is None:
                    continue

                if name not in self.delta:
                    self.delta[name] = torch.zeros_like(param.data)

                grad = param.grad
                norm = torch.norm(grad)

                if norm != 0:
                    self.delta[name] += self.epsilon * grad / (norm + EPS)
                    param.data.add_(self.delta[name])

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.delta:
                param.data.sub_(self.delta[name])

        self.delta.clear()


# =========================================================
# ADVERSARIAL TRAINING WRAPPER
# =========================================================

class AdversarialTrainer:
    """
    Unified adversarial training wrapper.

    Supports:
        - fgm
        - pgd
        - free
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = "fgm",
        epsilon: float = 1e-5,
        alpha: float = 1e-6,
        steps: int = 3,
        emb_name: str = "embedding",
    ) -> None:

        self.method = method

        if method == "fgm":
            self.strategy = FGM(model, epsilon, emb_name)

        elif method == "pgd":
            self.strategy = PGD(model, epsilon, alpha, steps, emb_name)

        elif method == "free":
            self.strategy = FreeAT(model, epsilon, emb_name)

        else:
            raise ValueError(f"Unsupported method: {method}")

    def attack(self, **kwargs) -> None:
        self.strategy.attack(**kwargs)

    def restore(self) -> None:
        self.strategy.restore()

    def backup_grad(self) -> None:
        if hasattr(self.strategy, "backup_grad"):
            self.strategy.backup_grad()

    def restore_grad(self) -> None:
        if hasattr(self.strategy, "restore_grad"):
            self.strategy.restore_grad()