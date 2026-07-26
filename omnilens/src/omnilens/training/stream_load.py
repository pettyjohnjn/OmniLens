# src/omnilens/training/stream_load.py
"""
Streaming FSDP loader for models too large to materialize on one node's CPU.

The stock multi-node path (``load_sharded_model_multinode``) calls
``from_pretrained(device_map="cpu")`` on *every* rank, so each rank holds the
whole model in CPU RAM before FSDP shards it. That is fine for 8B (16GB) but
impossible for 405B (812GB) on the cluster's 512GB nodes.

This loader never materializes the full model anywhere:

  1. Every rank builds the model on the ``meta`` device via
     ``init_empty_weights(include_buffers=False)`` -- params are meta (zero bytes)
     while buffers (rotary ``inv_freq``) are computed for real from config, so we
     never have to reconstruct them.
  2. The unembed (final norm + lm_head) is streamed from disk directly, before
     wrapping, so the lens side has real weights.
  3. FSDP wraps with ``param_init_fn``: for each wrapped unit, rank 0 streams that
     unit's still-meta parameters from the on-disk safetensors shards (one layer
     ~6.4GB at a time), other ranks allocate empty tensors, and
     ``sync_module_states=True`` broadcasts rank 0's real weights before
     FULL_SHARD splits them across ranks.

Peak footprint on any rank is one transformer layer plus its shard of the rest --
never the whole model. Only rank 0 reads bulk weights from disk.

Usage (see cli/main.py multinode branch)::

    model, index, qnames = build_meta_model(model_dir, dtype, attn_impl)
    materialize_params(model, index, ["model.norm.weight", "lm_head.weight"], dev, dtype)
    unembed = HFUnembed(model).clone_to_device(dev)   # real copy for the lens
    ... build model_cfg + activation_site_plan from `model` (structure only) ...
    fsdp_model, shard_state = wrap_fsdp_streaming(model, index, qnames,
                                                  local_rank=lr, dtype=dtype)
"""

from __future__ import annotations

import json
import logging
import os
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .model_shard import ModelShardState

logger = logging.getLogger(__name__)


def resolve_local_dir(model_name: str) -> str:
    if not os.path.isdir(model_name):
        raise FileNotFoundError(
            f"stream_load requires a local model directory, got {model_name!r}. "
            "Download/convert the checkpoint to disk first."
        )
    return model_name


class SafetensorsIndex:
    """Lazily reads tensors by name from a directory of safetensors shards.

    Caches one open handle per shard file. A single-file checkpoint
    (``model.safetensors`` with no index) is handled too.
    """

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        else:
            single = "model.safetensors"
            if not os.path.exists(os.path.join(model_dir, single)):
                raise FileNotFoundError(f"No safetensors index or {single} in {model_dir}")
            from safetensors import safe_open

            with safe_open(os.path.join(model_dir, single), framework="pt", device="cpu") as f:
                self.weight_map = {k: single for k in f.keys()}
        self._handles: Dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        shard = self.weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(os.path.join(self.model_dir, shard), framework="pt", device="cpu")
            self._handles[shard] = handle
        return handle.get_tensor(name)


def build_meta_model(
    model_name: str,
    dtype: torch.dtype,
    attn_implementation: Optional[str] = None,
) -> Tuple[nn.Module, SafetensorsIndex, Dict[int, str]]:
    """Instantiate the model on ``meta`` (params) with real buffers, plus a
    safetensors index and an ``id(module) -> qualified name`` map captured before
    FSDP mutates the tree."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    model_dir = resolve_local_dir(model_name)
    index = SafetensorsIndex(model_dir)

    cfg = AutoConfig.from_pretrained(model_dir)
    kwargs = {"dtype": dtype}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg, **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    qualified_names = {id(m): name for name, m in model.named_modules()}
    return model, index, qualified_names


def materialize_params(
    root: nn.Module,
    index: SafetensorsIndex,
    names: List[str],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Materialize specific parameters (by full dotted name) from disk into
    ``root``. Used for the unembed weights before wrapping."""
    from accelerate.utils import set_module_tensor_to_device

    for full in names:
        if not index.has(full):
            continue
        value = index.get(full).to(device=device, dtype=dtype)
        set_module_tensor_to_device(root, full, device, value=value)


def _materialize_subtree(
    qualified_names: Dict[int, str],
    index: SafetensorsIndex,
    module: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    *,
    load_real: bool,
) -> None:
    """FSDP ``param_init_fn``: fill every still-``meta`` parameter in ``module``.

    ``load_real`` -> copy the real weight from disk (rank 0). Otherwise allocate an
    empty tensor of the right shape (rank>0); ``sync_module_states`` overwrites it
    by broadcasting rank 0. Non-meta tensors (pre-materialized unembed, real
    buffers, already-wrapped children) are skipped.
    """
    from accelerate.utils import set_module_tensor_to_device

    prefix = qualified_names.get(id(module), "")
    for local_name, param in list(module.named_parameters()):
        if not param.is_meta:
            continue
        full = f"{prefix}.{local_name}" if prefix else local_name
        if load_real:
            if not index.has(full):
                raise KeyError(f"Parameter {full!r} not in checkpoint index")
            value = index.get(full).to(device=device, dtype=dtype)
        else:
            value = torch.empty(param.shape, dtype=dtype, device=device)
        set_module_tensor_to_device(module, local_name, device, value=value)


def wrap_fsdp_streaming(
    model: nn.Module,
    index: SafetensorsIndex,
    qualified_names: Dict[int, str],
    *,
    local_rank: int,
    dtype: torch.dtype = torch.bfloat16,
    cpu_offload: bool = False,
) -> Tuple[nn.Module, ModelShardState]:
    """Wrap a meta-initialized model with FSDP, streaming weights from disk.

    Must run after ``init_process_group``.
    """
    import torch.distributed as dist
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    from .model_shard import _get_transformer_layer_cls

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    layer_cls = _get_transformer_layer_cls(model)
    if layer_cls is None:
        raise RuntimeError("Could not detect transformer layer class for FSDP wrapping.")
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})

    param_init_fn = partial(
        _materialize_subtree,
        qualified_names,
        index,
        device=device,
        dtype=dtype,
        load_real=(rank == 0),
    )

    if rank == 0:
        logger.info(
            "[stream_load] rank0 streaming %d tensors across %d ranks (wrap=%s)",
            len(index.weight_map), world_size, layer_cls.__name__,
        )

    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "device_id": local_rank,
        "sync_module_states": True,   # broadcast rank-0's streamed weights
        "param_init_fn": param_init_fn,
        "forward_prefetch": True,
        "limit_all_gathers": True,
    }
    if cpu_offload:
        fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

    fsdp_model = FSDP(model, **fsdp_kwargs)

    if rank == 0:
        shard_gb = sum(p.numel() * p.element_size() for p in fsdp_model.parameters()) / 1e9
        logger.info("[stream_load] FSDP wrap done. Per-rank shard ~%.1fGB", shard_gb)

    shard_state = ModelShardState(
        enabled=True, lens_device=device, device_map=None, num_gpus=world_size, model_dtype=dtype
    )
    return fsdp_model, shard_state
