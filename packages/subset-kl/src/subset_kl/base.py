# src/subset_kl/base.py
"""Base class and utilities for loss functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Literal

import torch


ReductionType = Literal["none", "mean", "sum"]


def apply_reduction(
    loss: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = "mean",
) -> torch.Tensor:
    """
    Apply reduction to per-token loss.
    
    Parameters
    ----------
    loss : torch.Tensor
        Per-token loss of shape [batch, seq] or [N].
    mask : Optional[torch.Tensor]
        Attention mask of same shape as loss.
    reduction : str
        "none", "mean", or "sum".
        
    Returns
    -------
    torch.Tensor
        Reduced loss.
    """
    if reduction == "none":
        if mask is not None:
            return loss * mask.to(loss.dtype)
        return loss
        
    if mask is not None:
        loss = loss * mask.to(loss.dtype)
        
    if reduction == "sum":
        return loss.sum()
        
    if reduction == "mean":
        if mask is not None:
            denom = mask.sum().clamp_min(1.0)
        else:
            denom = loss.numel()
        return loss.sum() / denom
        
    raise ValueError(f"Unknown reduction: {reduction}")


class BaseLoss(ABC):
    """
    Abstract base class for KL divergence losses.
    
    All loss functions take student logits, teacher logits,
    and an optional attention mask, returning a scalar (or per-token) loss.
    
    Parameters
    ----------
    reduction : str
        How to reduce the loss: "none", "mean", or "sum".
    """

    def __init__(self, reduction: ReductionType = "mean") -> None:
        self.reduction = reduction

    @abstractmethod
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the loss.
        
        Parameters
        ----------
        student_logits : torch.Tensor
            Logits from the student, shape [batch, seq, vocab].
        teacher_logits : torch.Tensor
            Logits from the teacher, shape [batch, seq, vocab].
        attention_mask : Optional[torch.Tensor]
            Mask indicating valid tokens, shape [batch, seq].
            
        Returns
        -------
        torch.Tensor
            Loss value (scalar if reduction != "none").
        """
        raise NotImplementedError

    def __call__(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss (calls forward)."""
        return self.forward(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            attention_mask=attention_mask,
        )

    def _apply_reduction(
        self,
        loss: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply reduction to per-token loss."""
        return apply_reduction(loss, mask, self.reduction)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reduction={self.reduction!r})"
