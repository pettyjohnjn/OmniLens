# src/hookbox/base.py
"""Base classes for hooks."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch.utils.hooks import RemovableHandle


class BaseHook(ABC):
    """
    Abstract base class for all hooks.

    Subclasses must implement `register()` to attach the hook to a module.

    Attributes
    ----------
    name : str
        Unique identifier for this hook.
    module : Optional[nn.Module]
        The module this hook is attached to (None if not registered).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._handle: RemovableHandle | None = None
        self.module: nn.Module | None = None
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Number of times this hook has fired."""
        return self._call_count

    def reset_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0

    @abstractmethod
    def register(self, module: nn.Module) -> None:
        """
        Attach the hook to a module.

        Must set self._handle and self.module.

        Parameters
        ----------
        module : nn.Module
            The module to attach the hook to.
        """
        raise NotImplementedError

    def unregister(self) -> None:
        """Remove the hook from its module."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.module = None

    @property
    def is_registered(self) -> bool:
        """Whether this hook is currently attached to a module."""
        return self._handle is not None

    def __repr__(self) -> str:
        status = "registered" if self.is_registered else "unregistered"
        return f"{self.__class__.__name__}(name={self.name!r}, {status})"

    def __enter__(self) -> BaseHook:
        """Context manager entry (hook should already be registered)."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit - unregister the hook."""
        self.unregister()
