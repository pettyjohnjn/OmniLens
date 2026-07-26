# src/omnilens/__init__.py
"""OmniLens - Scalable lens-based interpretability for large language models."""

__version__ = "0.3.0"

from omnilens.hooks import ActivationCollector, HookManager
from omnilens.losses import create_loss, BaseLoss
from omnilens.lenses import create_lens, BaseLens, LogitLens, TunedLens, LowRankLens
from omnilens.training import LensTrainer, TrainConfig, HFUnembed, get_model_config

try:
    from omnilens.ops import indexed_logits, indexed_logits_available
except ImportError:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "omnilens.ops (CUDA extension for indexed_logits) is not available. "
        "Falling back to torch.gather. "
        "To build the extension: cd src/omnilens/ops && python setup.py build_ext --inplace"
    )
    indexed_logits = None
    indexed_logits_available = lambda: False

__all__ = [
    "__version__",
    "ActivationCollector",
    "HookManager",
    "create_loss",
    "BaseLoss",
    "create_lens",
    "BaseLens",
    "LogitLens",
    "TunedLens",
    "LowRankLens",
    "indexed_logits",
    "indexed_logits_available",
    "LensTrainer",
    "TrainConfig",
    "HFUnembed",
    "get_model_config",
]
