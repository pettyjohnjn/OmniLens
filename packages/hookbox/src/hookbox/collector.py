# src/hookbox/collector.py
"""
High-level activation collection interface.

This module provides the main user-facing API for collecting activations
from transformer models. It can be used as a context manager for clean
resource management.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .activation_hook import ActivationHook, InputHook
from .distributed import (
    DistributedInfo,
    gather_tensors,
    get_distributed_info,
    get_module_by_name,
    unwrap_model,
)
from .manager import HookManager


@dataclass
class CollectedActivations:
    """
    Container for collected activations from a forward pass.

    `hidden_states` includes embeddings at index 0. `custom` contains any
    additional activations captured via hooks.
    """
    hidden_states: tuple[torch.Tensor, ...] = field(default_factory=tuple)
    logits: torch.Tensor | None = None
    custom: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def num_layers(self) -> int:
        """Number of transformer layers (excluding embeddings)."""
        return len(self.hidden_states) - 1 if self.hidden_states else 0

    @property
    def embeddings(self) -> torch.Tensor | None:
        """Embedding layer output (index 0 of hidden_states)."""
        return self.hidden_states[0] if self.hidden_states else None

    @property
    def last_hidden_state(self) -> torch.Tensor | None:
        """Final layer hidden state."""
        return self.hidden_states[-1] if self.hidden_states else None

    def get_layer(self, layer_id: int | str) -> torch.Tensor:
        """
        Get hidden states for a specific layer.

        Parameters
        ----------
        layer_id : int or str
            Layer index (0-based, after embeddings).

        Returns
        -------
        torch.Tensor
            Hidden states of shape [batch, seq, hidden].

        Raises
        ------
        IndexError
            If layer_id is out of range.
        """
        idx = int(layer_id) + 1  # +1 to skip embeddings
        if idx >= len(self.hidden_states):
            raise IndexError(
                f"Layer {layer_id} out of range. "
                f"Model has {self.num_layers} layers (0-{self.num_layers-1})."
            )
        return self.hidden_states[idx]

    def iter_layers(self) -> Iterator[tuple[int, torch.Tensor]]:
        """
        Iterate over layers (excluding embeddings).

        Yields
        ------
        Tuple[int, torch.Tensor]
            (layer_id, hidden_states) pairs.
        """
        yield from enumerate(self.hidden_states[1:])

    def _map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> CollectedActivations:
        """Apply ``fn`` to every tensor, returning a new instance."""
        return CollectedActivations(
            hidden_states=tuple(fn(h) for h in self.hidden_states),
            logits=fn(self.logits) if self.logits is not None else None,
            custom={k: fn(v) for k, v in self.custom.items()},
        )

    def to(self, device: str | torch.device) -> CollectedActivations:
        """Move all tensors to a device (returns a new instance)."""
        return self._map(lambda t: t.to(device))

    def detach(self) -> CollectedActivations:
        """Detach all tensors from the computation graph (returns a new instance)."""
        return self._map(lambda t: t.detach())

    def clone(self) -> CollectedActivations:
        """Clone all tensors (returns a new instance)."""
        return self._map(lambda t: t.clone())

    def cpu(self) -> CollectedActivations:
        """Move all tensors to CPU."""
        return self.to("cpu")

    def cuda(self, device: int | None = None) -> CollectedActivations:
        """Move all tensors to CUDA."""
        return self.to(f"cuda:{device}" if device is not None else "cuda")

    def half(self) -> CollectedActivations:
        """Convert all tensors to float16."""
        return self._map(lambda t: t.half())

    def float(self) -> CollectedActivations:
        """Convert all tensors to float32."""
        return self._map(lambda t: t.float())

    def __repr__(self) -> str:
        shape_str = ""
        if self.hidden_states:
            shape_str = f", shape={list(self.hidden_states[0].shape)}"
        return (
            f"CollectedActivations(num_layers={self.num_layers}{shape_str}, "
            f"has_logits={self.logits is not None}, "
            f"custom_keys={list(self.custom.keys())})"
        )


class ActivationCollector:
    """
    High-level interface for collecting activations from a model.

    Works as a context manager or via attach()/detach().
    Hidden states are collected via `output_hidden_states=True`;
    custom hooks capture anything additional.
    """

    def __init__(
        self,
        model: nn.Module,
        custom_hooks: dict[str, str] | None = None,
        gather_distributed: bool = False,
        target_device: str | torch.device | None = None,
    ) -> None:
        self._wrapped_model = model
        self.custom_hooks = custom_hooks or {}
        self.gather_distributed = gather_distributed
        self.target_device = torch.device(target_device) if target_device else None

        self._manager = HookManager(model, auto_unwrap=True)
        self._custom_activations: dict[str, torch.Tensor] = {}
        self._attached = False

        # Cache distributed info
        self._dist_info: DistributedInfo | None = None

    @property
    def model(self) -> nn.Module:
        """The unwrapped model."""
        return unwrap_model(self._wrapped_model)

    @property
    def distributed_info(self) -> DistributedInfo:
        """Information about distributed setup."""
        if self._dist_info is None:
            self._dist_info = get_distributed_info(self._wrapped_model)
        return self._dist_info

    def attach(self) -> ActivationCollector:
        """
        Attach custom hooks to the model.

        Note: Hidden states are collected via output_hidden_states=True,
        not via hooks, for efficiency.

        Returns
        -------
        ActivationCollector
            Self, for method chaining.
        """
        if self._attached:
            return self

        # Add custom hooks
        for name, module_path in self.custom_hooks.items():
            def make_callback(capture_name: str):
                def callback(x, module_name, module):
                    self._custom_activations[capture_name] = x
                return callback

            if name.endswith("_input"):
                hook = InputHook(
                    name=f"collector:{name}",
                    on_input=make_callback(name),
                    module_name=module_path,
                )
            else:
                hook = ActivationHook(
                    name=f"collector:{name}",
                    on_activation=make_callback(name),
                    module_name=module_path,
                )
            self._manager.add_hook(module_path, hook)

        self._attached = True
        return self

    def detach(self) -> None:
        """Remove all hooks from the model."""
        self._manager.remove_all()
        self._custom_activations.clear()
        self._attached = False

    def _align_custom_with_hidden_states(
        self,
        hidden_states: tuple[torch.Tensor, ...] | None,
        custom: dict[str, torch.Tensor],
        *,
        detach: bool,
    ) -> dict[str, torch.Tensor]:
        if not hidden_states:
            return custom

        for name, module_path in self.custom_hooks.items():
            if name.endswith("_input"):
                continue

            parts = module_path.split(".")
            if len(parts) < 2:
                continue
            if not parts[-1].isdigit():
                continue

            idx = int(parts[-1])
            parent_path = ".".join(parts[:-1])
            try:
                parent = get_module_by_name(
                    self._wrapped_model,
                    parent_path,
                    unwrap=True,
                )
            except ValueError:
                continue

            if not isinstance(parent, nn.ModuleList):
                continue
            if len(parent) + 1 != len(hidden_states):
                continue
            if not (0 <= idx < len(parent)):
                continue

            tensor = hidden_states[idx + 1]
            custom[name] = tensor.detach() if detach else tensor

        return custom

    def _postprocess(self, data: CollectedActivations) -> CollectedActivations:
        """Apply distributed gathering and device transfer if configured."""
        if self.gather_distributed and self.distributed_info.is_distributed:
            data = data._map(lambda t: gather_tensors(t, dim=0))
        if self.target_device is not None:
            data = data.to(self.target_device)
        return data

    def _collect(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        grad: bool,
        **model_kwargs,
    ) -> CollectedActivations:
        """Run a forward pass and assemble the collected activations.

        When ``grad`` is False the forward runs under ``torch.no_grad()`` and the
        result is post-processed (gather/device). The grad path skips post-
        processing, since that can break the autograd graph — the caller handles
        device/gathering if needed.
        """
        self._custom_activations.clear()

        with nullcontext() if grad else torch.no_grad():
            outputs = self._wrapped_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
                **model_kwargs,
            )

        data = CollectedActivations(
            hidden_states=outputs.hidden_states,
            logits=getattr(outputs, "logits", None),
            custom=self._align_custom_with_hidden_states(
                outputs.hidden_states,
                dict(self._custom_activations),
                detach=not grad,
            ),
        )

        return data if grad else self._postprocess(data)

    def collect(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **model_kwargs,
    ) -> CollectedActivations:
        """Run a forward pass and collect activations (no gradients).

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape [batch, seq].
        attention_mask : torch.Tensor | None
            Attention mask of shape [batch, seq].
        **model_kwargs
            Additional arguments passed to ``model.forward()``.

        Returns
        -------
        CollectedActivations
            Container with hidden states, logits, and custom activations.
        """
        return self._collect(input_ids, attention_mask, grad=False, **model_kwargs)

    def collect_with_grad(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **model_kwargs,
    ) -> CollectedActivations:
        """Run a forward pass and collect activations WITH gradients.

        Use this during training when you need gradients to flow through the
        activations. Unlike :meth:`collect`, the result is not post-processed
        (gather/device), to avoid breaking the autograd graph.
        """
        return self._collect(input_ids, attention_mask, grad=True, **model_kwargs)

    @property
    def is_attached(self) -> bool:
        """Whether the collector is currently attached to the model."""
        return self._attached

    def __enter__(self) -> ActivationCollector:
        """Context manager entry."""
        return self.attach()

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.detach()

    def __repr__(self) -> str:
        status = "attached" if self._attached else "detached"
        model_name = self.model.__class__.__name__
        dist_str = ""
        if self.distributed_info.is_distributed:
            dist_str = f", {self.distributed_info.wrapper_type.name}"
        return (
            f"ActivationCollector(model={model_name}{dist_str}, "
            f"custom_hooks={len(self.custom_hooks)}, {status})"
        )
