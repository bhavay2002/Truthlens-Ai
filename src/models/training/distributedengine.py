from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class DistributedConfig:
    backend: str = "nccl"  # "nccl" | "gloo"
    init_method: str = "env://"
    use_ddp: bool = True
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True


# =========================================================
# DISTRIBUTED ENGINE
# =========================================================

class DistributedEngine:
    """
    Distributed training engine (DDP-ready).

    Responsibilities:
    - initialize process group
    - wrap model with DDP
    - manage rank/world_size
    - provide distributed utilities
    """

    def __init__(self, config: Optional[DistributedConfig] = None):

        self.config = config or DistributedConfig()

        self.initialized = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0

    # =====================================================
    # INIT
    # =====================================================

    def initialize(self):

        if dist.is_available() and not dist.is_initialized():

            self.rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

            logger.info(
                "Initializing distributed | rank=%d | world_size=%d",
                self.rank,
                self.world_size,
            )

            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                rank=self.rank,
                world_size=self.world_size,
            )

            torch.cuda.set_device(self.local_rank)

            self.initialized = True

        else:
            logger.info("Distributed not initialized (single process)")

    # =====================================================
    # MODEL WRAP
    # =====================================================

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:

        if not self.initialized or not self.config.use_ddp:
            return model

        device = torch.device(f"cuda:{self.local_rank}")

        model = model.to(device)

        model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=self.config.find_unused_parameters,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
        )

        logger.info("Model wrapped with DDP")

        return model

    # =====================================================
    # SAMPLER
    # =====================================================

    def create_sampler(self, dataset, shuffle: bool = True):

        if not self.initialized:
            return None

        return torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
        )

    # =====================================================
    # SYNC UTILS
    # =====================================================

    def barrier(self):
        if self.initialized:
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:

        if not self.initialized:
            return tensor

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / self.world_size

        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0):

        if self.initialized:
            dist.broadcast(tensor, src)

    # =====================================================
    # HELPERS
    # =====================================================

    def is_main_process(self) -> bool:
        return self.rank == 0

    def cleanup(self):

        if self.initialized:
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")