# src/hookbox/__init__.py
"""
hookbox: activation capture and transformation utilities for PyTorch models.
"""

__version__ = "0.1.0"

from .activation_hook import (
    ActivationHook,
    CallbackFn,
    GradientHook,
    InputHook,
    TensorHook,
    TransformFn,
)
from .base import BaseHook
from .collector import ActivationCollector, CollectedActivations
from .distributed import (
    # Checkpointing
    CheckpointingState,
    DistributedInfo,
    # Wrapper detection
    WrapperType,
    detect_wrapper_type,
    # Tensor operations
    gather_tensors,
    get_distributed_info,
    get_module_by_name,
    is_checkpointing_recomputation,
    # Unwrapping
    unwrap_model,
)
from .manager import HookFactory, HookManager, ModulePredicate

__all__ = [
    # Version
    "__version__",
    # Base
    "BaseHook",
    # Hooks
    "ActivationHook",
    "InputHook",
    "GradientHook",
    "TensorHook",
    # Manager
    "HookManager",
    # Collector
    "ActivationCollector",
    "CollectedActivations",
    # Distributed
    "WrapperType",
    "DistributedInfo",
    "detect_wrapper_type",
    "get_distributed_info",
    "unwrap_model",
    "get_module_by_name",
    "gather_tensors",
    "CheckpointingState",
    "is_checkpointing_recomputation",
    # Type aliases
    "TransformFn",
    "CallbackFn",
    "ModulePredicate",
    "HookFactory",
]
