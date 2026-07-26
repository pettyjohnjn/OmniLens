# src/hookbox/manager.py
"""Hook manager for attaching/detaching multiple hooks."""

from __future__ import annotations

from typing import Callable, Iterator

import torch.nn as nn

from .activation_hook import ActivationHook, CallbackFn, GradientHook, TransformFn
from .base import BaseHook
from .distributed import get_module_by_name, unwrap_model

# Type aliases
ModulePredicate = Callable[[str, nn.Module], bool]
HookFactory = Callable[[str, nn.Module], BaseHook]


class HookManager:
    """
    Manages multiple hooks attached to a model.

    Methods support attaching hooks by name or predicate and cleaning them up.
    Handles wrapped models (DDP/FSDP/DeepSpeed) by default via auto-unwrapping.
    """

    def __init__(self, model: nn.Module, auto_unwrap: bool = True) -> None:
        self._wrapped_model = model
        self._auto_unwrap = auto_unwrap
        self._hooks: dict[str, BaseHook] = {}

    @property
    def model(self) -> nn.Module:
        """The model (unwrapped if auto_unwrap=True)."""
        if self._auto_unwrap:
            return unwrap_model(self._wrapped_model)
        return self._wrapped_model

    @property
    def wrapped_model(self) -> nn.Module:
        """The original wrapped model."""
        return self._wrapped_model

    def add_hook(self, module_name: str, hook: BaseHook) -> None:
        """
        Attach a hook to a named module.

        Parameters
        ----------
        module_name : str
            Dot-separated path to module (e.g., "transformer.h.0.mlp").
        hook : BaseHook
            Hook instance to attach.

        Raises
        ------
        ValueError
            If module not found or hook name already registered.

        Examples
        --------
        >>> hook = ActivationHook(name="mlp_0", on_activation=callback)
        >>> manager.add_hook("transformer.h.0.mlp", hook)
        """
        # Find module (handles unwrapping internally)
        module = get_module_by_name(
            self._wrapped_model,
            module_name,
            unwrap=self._auto_unwrap
        )

        # Check for name collision
        if hook.name in self._hooks:
            raise ValueError(
                f"Hook '{hook.name}' already registered. "
                f"Use a unique name or remove the existing hook first."
            )

        # Set module_name on hook if it supports it
        if hasattr(hook, "module_name") and hook.module_name is None:
            hook.module_name = module_name

        # Register and track
        hook.register(module)
        self._hooks[hook.name] = hook

    def add_hooks_by_predicate(
        self,
        predicate: ModulePredicate,
        hook_factory: HookFactory,
    ) -> int:
        """
        Attach hooks to all modules matching a predicate.

        Parameters
        ----------
        predicate : Callable[[str, nn.Module], bool]
            Function that returns True for modules that should get hooks.
        hook_factory : Callable[[str, nn.Module], BaseHook]
            Function that creates a hook for a given module.

        Returns
        -------
        int
            Number of hooks added.

        Examples
        --------
        >>> # Hook all MLP layers
        >>> count = manager.add_hooks_by_predicate(
        ...     predicate=lambda name, mod: "mlp" in name,
        ...     hook_factory=lambda name, mod: ActivationHook(
        ...         name=f"mlp:{name}",
        ...         on_activation=callback
        ...     )
        ... )
        """
        def has_forward(mod: nn.Module) -> bool:
            return type(mod).forward is not nn.Module.forward

        count = 0
        for name, module in self.model.named_modules():
            if not has_forward(module):
                continue
            if predicate(name, module):
                hook = hook_factory(name, module)

                if hasattr(hook, "module_name") and hook.module_name is None:
                    hook.module_name = name

                hook.register(module)

                if hook.name in self._hooks:
                    hook.unregister()
                    raise ValueError(
                        f"Hook '{hook.name}' already registered. "
                        f"Ensure hook_factory generates unique names."
                    )
                self._hooks[hook.name] = hook
                count += 1

        return count

    def add_activation_hooks(
        self,
        predicate: ModulePredicate,
        *,
        on_activation: CallbackFn | None = None,
        transform_fn: TransformFn | None = None,
        name_prefix: str = "activation",
        skip_checkpointing_recompute: bool = True,
    ) -> int:
        """
        Convenience method to add ActivationHooks to matching modules.

        Parameters
        ----------
        predicate : ModulePredicate
            Function that selects which modules get hooks.
        on_activation : Optional[CallbackFn]
            Callback for each activation.
        transform_fn : Optional[TransformFn]
            Transform function for interventions.
        name_prefix : str
            Prefix for generated hook names.
        skip_checkpointing_recompute : bool
            Skip callback during activation checkpointing recomputation.

        Returns
        -------
        int
            Number of hooks added.

        Examples
        --------
        >>> # Capture all layer outputs
        >>> activations = []
        >>> manager.add_activation_hooks(
        ...     predicate=lambda n, m: n.endswith(".output"),
        ...     on_activation=lambda x, n, m: activations.append(x.clone()),
        ... )
        """
        def factory(module_name: str, module: nn.Module) -> BaseHook:
            return ActivationHook(
                name=f"{name_prefix}:{module_name}",
                on_activation=on_activation,
                transform_fn=transform_fn,
                module_name=module_name,
                skip_checkpointing_recompute=skip_checkpointing_recompute,
            )

        return self.add_hooks_by_predicate(predicate, factory)

    def add_gradient_hooks(
        self,
        predicate: ModulePredicate,
        *,
        on_gradient: Callable | None = None,
        transform_fn: Callable | None = None,
        name_prefix: str = "gradient",
    ) -> int:
        """
        Convenience method to add GradientHooks to matching modules.

        Parameters
        ----------
        predicate : ModulePredicate
            Function that selects which modules get hooks.
        on_gradient : Optional[Callable]
            Callback for each gradient.
        transform_fn : Optional[Callable]
            Transform function for gradient modification.
        name_prefix : str
            Prefix for generated hook names.

        Returns
        -------
        int
            Number of hooks added.
        """
        def factory(module_name: str, module: nn.Module) -> BaseHook:
            return GradientHook(
                name=f"{name_prefix}:{module_name}",
                on_gradient=on_gradient,
                transform_fn=transform_fn,
                module_name=module_name,
            )

        return self.add_hooks_by_predicate(predicate, factory)

    def remove_hook(self, name: str) -> bool:
        """
        Remove a specific hook by name.

        Parameters
        ----------
        name : str
            Name of the hook to remove.

        Returns
        -------
        bool
            True if hook was found and removed, False otherwise.
        """
        hook = self._hooks.pop(name, None)
        if hook is not None:
            hook.unregister()
            return True
        return False

    def remove_all(self) -> int:
        """
        Remove all registered hooks.

        Returns
        -------
        int
            Number of hooks removed.
        """
        count = len(self._hooks)
        for hook in self._hooks.values():
            hook.unregister()
        self._hooks.clear()
        return count

    def get_hook(self, name: str) -> BaseHook | None:
        """Get a hook by name, or None if not found."""
        return self._hooks.get(name)

    def list_hooks(self) -> list[str]:
        """Return list of registered hook names."""
        return list(self._hooks.keys())

    def iter_hooks(self) -> Iterator[tuple[str, BaseHook]]:
        """Iterate over (name, hook) pairs."""
        yield from self._hooks.items()

    def reset_all_counts(self) -> None:
        """Reset call counts on all hooks that support it."""
        for hook in self._hooks.values():
            if hasattr(hook, "reset_count"):
                hook.reset_count()

    @property
    def num_hooks(self) -> int:
        """Number of registered hooks."""
        return len(self._hooks)

    def __contains__(self, name: str) -> bool:
        """Check if a hook with the given name is registered."""
        return name in self._hooks

    def __len__(self) -> int:
        """Number of registered hooks."""
        return len(self._hooks)

    def __enter__(self) -> HookManager:
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit - remove all hooks."""
        self.remove_all()

    def __repr__(self) -> str:
        model_name = self.model.__class__.__name__
        wrapper_name = self._wrapped_model.__class__.__name__
        if model_name != wrapper_name:
            model_str = f"{wrapper_name}({model_name})"
        else:
            model_str = model_name
        return f"HookManager(model={model_str}, hooks={self.num_hooks})"
