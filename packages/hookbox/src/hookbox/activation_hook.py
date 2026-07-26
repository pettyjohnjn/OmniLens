# src/hookbox/activation_hook.py
"""Activation capture and transform hooks."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from .base import BaseHook
from .distributed import is_checkpointing_recomputation

# Type aliases for callbacks
TransformFn = Callable[[torch.Tensor, str, nn.Module], torch.Tensor]
CallbackFn = Callable[[torch.Tensor, str, nn.Module], Any]


class ActivationHook(BaseHook):
    """
    Hook for capturing and optionally transforming activations during forward.

    If `on_activation` is set, it is called with (activation, module_name, module).
    If `transform_fn` is set, its return value replaces the activation.

    Notes:
    - Works with wrapped models (DDP/FSDP/etc.) and normal modules.
    - During activation checkpointing recompute, callbacks can be skipped.
    """

    def __init__(
        self,
        name: str,
        on_activation: CallbackFn | None = None,
        transform_fn: TransformFn | None = None,
        module_name: str | None = None,
        skip_checkpointing_recompute: bool = True,
    ) -> None:
        super().__init__(name)
        self.on_activation = on_activation
        self.transform_fn = transform_fn
        self.module_name = module_name
        self.skip_checkpointing_recompute = skip_checkpointing_recompute

    def register(self, module: nn.Module) -> None:
        """
        Attach hook to module's forward pass.

        Parameters
        ----------
        module : nn.Module
            The module to attach the hook to.
        """
        module_name = self.module_name or ""

        def hook_fn(
            mod: nn.Module,
            inputs: tuple,
            outputs: torch.Tensor | tuple
        ) -> torch.Tensor | tuple | None:
            self._call_count += 1

            # Check if we should skip callback during checkpointing recompute
            skip_callback = (
                self.skip_checkpointing_recompute
                and is_checkpointing_recomputation()
            )

            # Handle tensor outputs (simple case)
            if isinstance(outputs, torch.Tensor):
                if self.on_activation is not None and not skip_callback:
                    self.on_activation(outputs, module_name, mod)
                if self.transform_fn is not None:
                    return self.transform_fn(outputs, module_name, mod)
                return None  # Return None to not modify output

            # Handle tuple outputs (common in transformers - hidden_states, attentions, etc.)
            if isinstance(outputs, tuple) and len(outputs) > 0:
                first = outputs[0]
                if isinstance(first, torch.Tensor):
                    if self.on_activation is not None and not skip_callback:
                        self.on_activation(first, module_name, mod)
                    if self.transform_fn is not None:
                        transformed = self.transform_fn(first, module_name, mod)
                        return (transformed,) + outputs[1:]
                return None

            # Non-tensor outputs: just callback, no transform possible
            if self.on_activation is not None and not skip_callback:
                self.on_activation(outputs, module_name, mod)
            return None

        self._handle = module.register_forward_hook(hook_fn)
        self.module = module

    def __repr__(self) -> str:
        status = "registered" if self.is_registered else "unregistered"
        parts = [f"name={self.name!r}"]
        if self.module_name:
            parts.append(f"module_name={self.module_name!r}")
        if self.on_activation is not None:
            parts.append("captures=True")
        if self.transform_fn is not None:
            parts.append("transforms=True")
        parts.append(f"calls={self._call_count}")
        parts.append(status)
        return f"ActivationHook({', '.join(parts)})"


class InputHook(BaseHook):
    """
    Hook for capturing module inputs via a forward pre-hook.
    """

    def __init__(
        self,
        name: str,
        on_input: CallbackFn | None = None,
        module_name: str | None = None,
        skip_checkpointing_recompute: bool = True,
    ) -> None:
        super().__init__(name)
        self.on_input = on_input
        self.module_name = module_name
        self.skip_checkpointing_recompute = skip_checkpointing_recompute

    def register(self, module: nn.Module) -> None:
        """Attach hook to module's forward pre-pass."""
        module_name = self.module_name or ""

        def hook_fn(mod: nn.Module, inputs: tuple):
            self._call_count += 1

            skip_callback = (
                self.skip_checkpointing_recompute
                and is_checkpointing_recomputation()
            )

            if self.on_input is None or skip_callback:
                return None

            # Inputs are always a tuple for forward_pre_hook
            if isinstance(inputs, tuple) and len(inputs) > 0:
                first = inputs[0]
                if isinstance(first, torch.Tensor):
                    self.on_input(first, module_name, mod)
                    return None
            # Fallback: pass through raw inputs if not a tensor
            self.on_input(inputs, module_name, mod)
            return None

        self._handle = module.register_forward_pre_hook(hook_fn)
        self.module = module

    def __repr__(self) -> str:
        status = "registered" if self.is_registered else "unregistered"
        parts = [f"name={self.name!r}"]
        if self.module_name:
            parts.append(f"module_name={self.module_name!r}")
        if self.on_input is not None:
            parts.append("captures=True")
        parts.append(f"calls={self._call_count}")
        parts.append(status)
        return f"InputHook({', '.join(parts)})"


class GradientHook(BaseHook):
    """
    Hook for capturing gradients during backward pass.
    """

    def __init__(
        self,
        name: str,
        on_gradient: Callable | None = None,
        transform_fn: Callable | None = None,
        module_name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.on_gradient = on_gradient
        self.transform_fn = transform_fn
        self.module_name = module_name

    def register(self, module: nn.Module) -> None:
        """Attach hook to module's backward pass."""
        module_name = self.module_name or ""

        def hook_fn(mod: nn.Module, grad_input: tuple, grad_output: tuple):
            self._call_count += 1

            if grad_output and grad_output[0] is not None:
                grad = grad_output[0]
                if self.on_gradient is not None:
                    self.on_gradient(grad, module_name, mod)
                if self.transform_fn is not None:
                    transformed = self.transform_fn(grad, module_name, mod)
                    return (transformed,) + grad_output[1:] if len(grad_output) > 1 else (transformed,)
            return None

        self._handle = module.register_full_backward_hook(hook_fn)
        self.module = module

    def __repr__(self) -> str:
        status = "registered" if self.is_registered else "unregistered"
        return (
            f"GradientHook(name={self.name!r}, "
            f"module_name={self.module_name!r}, "
            f"calls={self._call_count}, {status})"
        )


class TensorHook:
    """
    Hook that attaches directly to a tensor (not a module).

    Useful for capturing/transforming gradients of specific tensors
    during backward pass.

    Parameters
    ----------
    name : str
        Unique identifier for this hook.
    on_grad : Optional[Callable[[torch.Tensor], Any]]
        Callback invoked with the gradient tensor.
    transform_fn : Optional[Callable[[torch.Tensor], torch.Tensor]]
        Function to transform the gradient.

    """

    def __init__(
        self,
        name: str,
        on_grad: Callable[[torch.Tensor], Any] | None = None,
        transform_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.name = name
        self.on_grad = on_grad
        self.transform_fn = transform_fn
        self._handle = None
        self._tensor = None

    def register(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Attach hook to a tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to attach to. Must have requires_grad=True.

        Returns
        -------
        torch.Tensor
            The same tensor (for chaining).
        """
        if not tensor.requires_grad:
            raise ValueError(
                "Cannot attach gradient hook to tensor with requires_grad=False"
            )

        def hook_fn(grad: torch.Tensor) -> torch.Tensor | None:
            if self.on_grad is not None:
                self.on_grad(grad)
            if self.transform_fn is not None:
                return self.transform_fn(grad)
            return None

        self._handle = tensor.register_hook(hook_fn)
        self._tensor = tensor
        return tensor

    def unregister(self) -> None:
        """Remove the hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._tensor = None

    @property
    def is_registered(self) -> bool:
        """Whether this hook is currently attached."""
        return self._handle is not None

    def __repr__(self) -> str:
        status = "registered" if self.is_registered else "unregistered"
        return f"TensorHook(name={self.name!r}, {status})"
