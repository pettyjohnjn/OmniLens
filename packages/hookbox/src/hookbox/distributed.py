# src/hookbox/distributed.py
"""
Utilities for working with distributed and wrapped models.

This module provides helpers for:
- Detecting wrapped models (DDP, FSDP, DeepSpeed, etc.)
- Unwrapping to access the base model
- Device-aware activation gathering
- Handling activation checkpointing
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto

import torch
import torch.nn as nn


class WrapperType(Enum):
    """Types of model wrappers we can detect."""
    NONE = auto()
    DDP = auto()           # DistributedDataParallel
    FSDP = auto()          # FullyShardedDataParallel
    DEEPSPEED = auto()     # DeepSpeed ZeRO
    UNKNOWN = auto()       # Some other wrapper (has a .module attribute)


@dataclass
class DistributedInfo:
    """Information about a model's distributed setup."""
    wrapper_type: WrapperType
    wrapper_class: str | None
    world_size: int
    rank: int
    local_rank: int
    device: torch.device
    is_sharded: bool  # Whether parameters are sharded (FSDP, DeepSpeed ZeRO-3)

    @property
    def is_distributed(self) -> bool:
        """Whether the model uses any distributed wrapper."""
        return self.wrapper_type != WrapperType.NONE

    @property
    def is_main_process(self) -> bool:
        """Whether this is rank 0."""
        return self.rank == 0


def detect_wrapper_type(model: nn.Module) -> WrapperType:
    """
    Detect what kind of distributed wrapper (if any) is applied to a model.

    Parameters
    ----------
    model : nn.Module
        The model to check.

    Returns
    -------
    WrapperType
        The detected wrapper type.
    """
    class_name = model.__class__.__name__
    module_name = model.__class__.__module__

    # Check for DDP
    if class_name == "DistributedDataParallel":
        return WrapperType.DDP

    # Check for FSDP (various versions)
    if "FullyShardedDataParallel" in class_name or "FSDP" in class_name:
        return WrapperType.FSDP

    # Check for DeepSpeed
    if "deepspeed" in module_name.lower() or class_name == "DeepSpeedEngine":
        return WrapperType.DEEPSPEED

    # Check for common wrappers by attribute
    if hasattr(model, "module"):
        # Generic wrapper with .module attribute
        return WrapperType.UNKNOWN

    return WrapperType.NONE


def get_distributed_info(model: nn.Module) -> DistributedInfo:
    """
    Get information about a model's distributed setup.

    Parameters
    ----------
    model : nn.Module
        The model to inspect.

    Returns
    -------
    DistributedInfo
        Information about the distributed configuration.
    """
    wrapper_type = detect_wrapper_type(model)

    # Get distributed state from torch.distributed if available
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world_size = 1
        rank = 0

    # Try to get local rank from environment or model
    local_rank = _get_local_rank(model)

    # Determine device
    device = _get_model_device(model)

    # Check if sharded
    is_sharded = wrapper_type in (WrapperType.FSDP, WrapperType.DEEPSPEED)

    return DistributedInfo(
        wrapper_type=wrapper_type,
        wrapper_class=model.__class__.__name__ if wrapper_type != WrapperType.NONE else None,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        device=device,
        is_sharded=is_sharded,
    )


def _get_local_rank(model: nn.Module) -> int:
    """Get local rank from various sources."""
    import os

    # Try environment variables (common in distributed launchers)
    for var in ["LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"]:
        if var in os.environ:
            return int(os.environ[var])

    # Try to get from torch.distributed
    if torch.distributed.is_initialized():
        try:
            # This works if using torch.distributed.launch with local_rank
            return torch.distributed.get_rank() % torch.cuda.device_count()
        except Exception:
            pass

    return 0


def _get_model_device(model: nn.Module) -> torch.device:
    """Get the device of a model (from first parameter)."""
    try:
        param = next(model.parameters())
        return param.device
    except StopIteration:
        return torch.device("cpu")


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Unwrap a model to get the base module.
    Handles DDP/FSDP/DeepSpeed and nested wrappers.
    """
    # Keep unwrapping until we hit the base
    while True:
        # Check common wrapper attributes
        if hasattr(model, "module"):
            model = model.module
        # DeepSpeed specific
        elif hasattr(model, "model") and model.__class__.__name__ == "DeepSpeedEngine":
            model = model.model
        else:
            break

    return model


def get_module_by_name(model: nn.Module, name: str, unwrap: bool = True) -> nn.Module:
    """
    Get a submodule by its dotted name, optionally unwrapping first.
    """
    if unwrap:
        model = unwrap_model(model)

    module = dict(model.named_modules()).get(name)
    if module is None:
        available = [n for n, _ in model.named_modules() if n][:20]
        raise ValueError(
            f"Module '{name}' not found. "
            f"Available modules (first 20): {available}"
        )
    return module


def gather_tensors(
    tensor: torch.Tensor,
    world_size: int | None = None,
    dim: int = 0,
) -> torch.Tensor:
    """
    Gather tensors from all ranks along a dimension.

    Two intended uses: ``dim=0`` collects each data-parallel rank's batch shard;
    ``dim=-1`` reconstructs an activation that is tensor-parallel-sharded across
    the hidden dimension (e.g. a column-parallel QKV/MLP output).

    Parameters
    ----------
    tensor : torch.Tensor
        Local tensor to gather.
    world_size : Optional[int]
        Number of processes. Inferred from torch.distributed if None.
    dim : int
        Dimension to concatenate along.

    Returns
    -------
    torch.Tensor
        Gathered tensor (concatenated from all ranks).

    Notes
    -----
    No-op if torch.distributed is not initialized or world_size=1.

    .. warning::
        The tensor-parallel reconstruction path (``dim=-1``) is **not validated**:
        no consumer currently runs the model under tensor parallelism, so this
        has not been exercised on real TP-sharded activations. Treat it as a
        primitive, not as turnkey TP support, until covered by a real TP run.
    """
    if not torch.distributed.is_initialized():
        return tensor

    if world_size is None:
        world_size = torch.distributed.get_world_size()

    if world_size == 1:
        return tensor

    # Create placeholder tensors for gathering
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor)

    return torch.cat(gathered, dim=dim)


class CheckpointingState:
    """
    Tracks whether we're in a checkpointing recomputation phase.

    Activation checkpointing causes hooks to fire twice: once in the
    forward pass, once during gradient checkpointing recomputation.
    This class helps detect and handle that.
    """

    _is_recomputing: bool = False

    @classmethod
    def is_recomputing(cls) -> bool:
        """Check if we're currently in a checkpointing recomputation."""
        return cls._is_recomputing

    @classmethod
    @contextmanager
    def recomputation_context(cls):
        """Context manager to mark recomputation phase."""
        old_state = cls._is_recomputing
        cls._is_recomputing = True
        try:
            yield
        finally:
            cls._is_recomputing = old_state


def is_checkpointing_recomputation() -> bool:
    """
    Whether we're in an activation-checkpointing recomputation phase.

    This reflects the explicit ``CheckpointingState`` flag only; wrap your
    checkpoint function with ``CheckpointingState.recomputation_context()`` to
    set it. Without that, this always returns False (there is no implicit
    auto-detection).
    """
    return CheckpointingState.is_recomputing()
