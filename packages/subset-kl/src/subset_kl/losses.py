# src/subset_kl/losses.py
"""
Class-based loss interfaces for subset KL divergence.

These wrap the functional interfaces in `core.py` for use in training loops
where you want a stateful loss object.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import BaseLoss, ReductionType
from .core import (
    TailProposalType,
    select_head_tail_indices,
    select_topk_indices,
    subset_k2_kl_from_gathered,
    subset_k3_kl_from_gathered,
    subset_kl_from_gathered,
    subset_is_kl_from_gathered,
    full_kl,
)


class SubsetKLLoss(BaseLoss):
    """
    Memory-efficient subset KL loss (class interface).
    
    This class provides two usage patterns:
    
    **Pattern 1: Full student logits (convenience, no memory savings)**
    
        >>> loss_fn = SubsetKLLoss(k=256)
        >>> loss = loss_fn(student_logits, teacher_logits)  # [B,T,V] inputs
    
    **Pattern 2: Pre-gathered tensors (memory-efficient)**
    
        >>> loss_fn = SubsetKLLoss(k=256)
        >>> indices, teacher_k = loss_fn.select_indices(teacher_logits)
        >>> student_k = your_model.forward_subset(hidden, indices)  # You compute this!
        >>> loss = loss_fn.forward_gathered(student_k, teacher_k)
    
    Parameters
    ----------
    k : int
        Number of top tokens to use.
    reduction : str
        "none", "mean", or "sum".
        
    Examples
    --------
    Memory-efficient usage with a lens:
    
        >>> loss_fn = SubsetKLLoss(k=256)
        >>> 
        >>> # Step 1: Get indices from teacher
        >>> indices, teacher_k = loss_fn.select_indices(teacher_logits)
        >>> 
        >>> # Step 2: Compute student logits only for those indices
        >>> # (Use indexed_logits or your lens's vocab_indices.)
        >>> student_k = lens.forward(hidden, vocab_indices=indices).logits
        >>> 
        >>> # Step 3: Compute KL
        >>> loss = loss_fn.forward_gathered(student_k, teacher_k, attention_mask)
    """

    def __init__(
        self,
        k: int = 256,
        reduction: ReductionType = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.k = k
        
        # Cache last indices for debugging/inspection
        self._last_indices: Optional[torch.Tensor] = None

    def select_indices(
        self,
        teacher_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k indices from teacher distribution.
        
        Parameters
        ----------
        teacher_logits : torch.Tensor
            Teacher logits [batch, seq, vocab].
            
        Returns
        -------
        indices : torch.Tensor
            Top-k indices [batch, seq, k].
        teacher_logits_k : torch.Tensor
            Teacher logits for those indices [batch, seq, k].
        """
        indices, teacher_k = select_topk_indices(teacher_logits, self.k)
        self._last_indices = indices
        return indices, teacher_k

    def forward_gathered(
        self,
        student_logits_k: torch.Tensor,
        teacher_logits_k: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute KL from pre-gathered subset logits.
        
        Parameters
        ----------
        student_logits_k : torch.Tensor
            Student logits for subset [batch, seq, k].
        teacher_logits_k : torch.Tensor
            Teacher logits for same subset [batch, seq, k].
        attention_mask : Optional[torch.Tensor]
            Mask [batch, seq].
            
        Returns
        -------
        torch.Tensor
            Loss value.
        """
        return subset_kl_from_gathered(
            student_logits_k, teacher_logits_k, attention_mask, self.reduction
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute subset KL from full logits (convenience, no memory savings).
        
        This requires full [B, T, V] student logits. For efficiency, use
        `select_indices()` + your model's subset forward + `forward_gathered()`.
        
        Parameters
        ----------
        student_logits : torch.Tensor
            Student logits [batch, seq, vocab].
        teacher_logits : torch.Tensor
            Teacher logits [batch, seq, vocab].
        attention_mask : Optional[torch.Tensor]
            Mask [batch, seq].
        """
        indices, teacher_k = self.select_indices(teacher_logits)
        student_k = torch.gather(student_logits, -1, indices)
        return self.forward_gathered(student_k, teacher_k, attention_mask)

    @property
    def last_indices(self) -> Optional[torch.Tensor]:
        """Last selected indices (for debugging)."""
        return self._last_indices

    def __repr__(self) -> str:
        return f"SubsetKLLoss(k={self.k}, reduction={self.reduction!r})"


class KLDivergenceLoss(BaseLoss):
    """
    Full-vocabulary KL divergence loss (baseline).
    
    Use this for comparison with subset KL, or when memory isn't a concern.
    
    Parameters
    ----------
    reduction : str
        "none", "mean", or "sum".
    temperature : float
        Temperature for softmax (1.0 = no scaling).
    chunk_size : Optional[int]
        If set, compute KL in chunks to reduce peak memory.
    """

    def __init__(
        self,
        reduction: ReductionType = "mean",
        temperature: float = 1.0,
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.chunk_size = chunk_size

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute full-vocabulary KL divergence."""
        if self.chunk_size is not None:
            return self._forward_chunked(student_logits, teacher_logits, attention_mask)
        
        return full_kl(
            student_logits, teacher_logits, attention_mask,
            self.reduction, self.temperature
        )

    def _forward_chunked(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Chunked computation for memory efficiency."""
        batch, seq, vocab = student_logits.shape
        chunk_size = self.chunk_size
        
        total_loss = torch.zeros((), device=student_logits.device, dtype=torch.float32)
        total_count = torch.zeros((), device=student_logits.device, dtype=torch.float32)
        
        for t0 in range(0, seq, chunk_size):
            t1 = min(t0 + chunk_size, seq)
            
            s_chunk = student_logits[:, t0:t1, :]
            t_chunk = teacher_logits[:, t0:t1, :]
            m_chunk = attention_mask[:, t0:t1] if attention_mask is not None else None
            
            # Apply temperature
            if self.temperature != 1.0:
                s_chunk = s_chunk / self.temperature
                t_chunk = t_chunk / self.temperature
            
            s_logprobs = F.log_softmax(s_chunk, dim=-1)
            t_logprobs = F.log_softmax(t_chunk, dim=-1)
            t_probs = t_logprobs.exp()
            
            kl_chunk = (t_probs * (t_logprobs - s_logprobs)).sum(dim=-1)
            
            if m_chunk is not None:
                kl_chunk = kl_chunk * m_chunk.to(kl_chunk.dtype)
                total_count += m_chunk.sum()
            else:
                total_count += kl_chunk.numel()
            
            total_loss += kl_chunk.sum()
        
        if self.reduction == "sum":
            return total_loss
        elif self.reduction == "mean":
            return total_loss / total_count.clamp_min(1.0)
        else:
            raise ValueError("Chunked KL only supports 'mean' or 'sum' reduction")

    def __repr__(self) -> str:
        parts = [f"reduction={self.reduction!r}"]
        if self.temperature != 1.0:
            parts.append(f"temperature={self.temperature}")
        if self.chunk_size is not None:
            parts.append(f"chunk_size={self.chunk_size}")
        return f"KLDivergenceLoss({', '.join(parts)})"


class SubsetK2KLLoss(BaseLoss):
    """
    Top-k head KL with an importance-sampled K2 penalty on the teacher tail.

    This follows the subset KL objective:

        KL_head + (1 - sg(P_head)) / k_tail * sum_i (log P(t_i) - log Q(t_i))^2

    The efficient path is ``select_indices()`` followed by a model/lens forward
    on those indices and then ``forward_gathered()``. For ``k_tail > 0``,
    ``forward_gathered()`` also needs the full-vocabulary student log
    normalizer so the tail term uses the true ``log Q(t_i)``.
    """

    def __init__(
        self,
        k_head: int = 256,
        k_tail: int = 256,
        tail_proposal: TailProposalType = "target",
        tail_proposal_alpha: float = 0.8,
        tail_proposal_tau: float = 0.7,
        reduction: ReductionType = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.k_head = k_head
        self.k_tail = k_tail
        self.tail_proposal = tail_proposal
        self.tail_proposal_alpha = tail_proposal_alpha
        self.tail_proposal_tau = tail_proposal_tau
        self._last_indices: Optional[torch.Tensor] = None
        self._last_p_head: Optional[torch.Tensor] = None
        self._last_tail_proposal_log_probs: Optional[torch.Tensor] = None

    def select_indices(
        self,
        teacher_logits: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select top-k head plus sampled tail indices.

        Returns selected indices, full teacher log-probabilities at those
        indices, and the full teacher probability mass of the head.
        """
        indices, teacher_log_probs_selected, p_head = select_head_tail_indices(
            teacher_logits,
            k_head=self.k_head,
            k_tail=self.k_tail,
            generator=generator,
        )
        self._last_indices = indices
        self._last_p_head = p_head
        return indices, teacher_log_probs_selected, p_head

    def forward_gathered(
        self,
        student_logits_selected: torch.Tensor,
        teacher_log_probs_selected: torch.Tensor,
        p_head: Optional[torch.Tensor] = None,
        student_log_normalizer: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute subset K2 KL from pre-gathered selected logits."""
        return subset_k2_kl_from_gathered(
            student_logits_selected,
            teacher_log_probs_selected,
            k_head=self.k_head,
            k_tail=self.k_tail,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
            attention_mask=attention_mask,
            reduction=self.reduction,
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute subset K2 KL from full logits as a convenience path."""
        indices, teacher_log_probs_selected, p_head = self.select_indices(teacher_logits)
        student_selected = torch.gather(student_logits, -1, indices)
        student_log_normalizer = torch.logsumexp(student_logits.float(), dim=-1)
        return self.forward_gathered(
            student_selected,
            teacher_log_probs_selected,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
            attention_mask=attention_mask,
        )

    @property
    def last_indices(self) -> Optional[torch.Tensor]:
        """Last selected indices."""
        return self._last_indices

    @property
    def last_p_head(self) -> Optional[torch.Tensor]:
        """Last teacher head mass."""
        return self._last_p_head

    def __repr__(self) -> str:
        return (
            f"SubsetK2KLLoss(k_head={self.k_head}, k_tail={self.k_tail}, "
            f"reduction={self.reduction!r})"
        )


class SubsetK3KLLoss(BaseLoss):
    """
    Top-k head KL with Schulman's K3 estimator on the teacher tail.

    This follows the subset KL objective:

        KL_head + (1 - sg(P_head)) / k_tail
        * sum_i (exp(log Q(t_i) - log P(t_i)) - 1 - (log Q(t_i) - log P(t_i)))

    The efficient path is ``select_indices()`` followed by a model/lens forward
    on those indices and then ``forward_gathered()``. For ``k_tail > 0``,
    ``forward_gathered()`` also needs the full-vocabulary student log
    normalizer so the tail term uses the true ``log Q(t_i)``.
    """

    def __init__(
        self,
        k_head: int = 256,
        k_tail: int = 256,
        tail_proposal: TailProposalType = "target",
        tail_proposal_alpha: float = 0.8,
        tail_proposal_tau: float = 0.7,
        reduction: ReductionType = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.k_head = k_head
        self.k_tail = k_tail
        self.tail_proposal = tail_proposal
        self.tail_proposal_alpha = tail_proposal_alpha
        self.tail_proposal_tau = tail_proposal_tau
        self._last_indices: Optional[torch.Tensor] = None
        self._last_p_head: Optional[torch.Tensor] = None
        self._last_tail_proposal_log_probs: Optional[torch.Tensor] = None

    def select_indices(
        self,
        teacher_logits: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select top-k head plus sampled tail indices.

        Returns selected indices, full teacher log-probabilities at those
        indices, and the full teacher probability mass of the head.
        """
        indices, teacher_log_probs_selected, p_head = select_head_tail_indices(
            teacher_logits,
            k_head=self.k_head,
            k_tail=self.k_tail,
            generator=generator,
        )
        self._last_indices = indices
        self._last_p_head = p_head
        return indices, teacher_log_probs_selected, p_head

    def forward_gathered(
        self,
        student_logits_selected: torch.Tensor,
        teacher_log_probs_selected: torch.Tensor,
        p_head: Optional[torch.Tensor] = None,
        student_log_normalizer: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute subset K3 KL from pre-gathered selected logits."""
        return subset_k3_kl_from_gathered(
            student_logits_selected,
            teacher_log_probs_selected,
            k_head=self.k_head,
            k_tail=self.k_tail,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
            attention_mask=attention_mask,
            reduction=self.reduction,
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute subset K3 KL from full logits as a convenience path."""
        indices, teacher_log_probs_selected, p_head = self.select_indices(teacher_logits)
        student_selected = torch.gather(student_logits, -1, indices)
        student_log_normalizer = torch.logsumexp(student_logits.float(), dim=-1)
        return self.forward_gathered(
            student_selected,
            teacher_log_probs_selected,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
            attention_mask=attention_mask,
        )

    @property
    def last_indices(self) -> Optional[torch.Tensor]:
        """Last selected indices."""
        return self._last_indices

    @property
    def last_p_head(self) -> Optional[torch.Tensor]:
        """Last teacher head mass."""
        return self._last_p_head

    def __repr__(self) -> str:
        return (
            f"SubsetK3KLLoss(k_head={self.k_head}, k_tail={self.k_tail}, "
            f"reduction={self.reduction!r})"
        )


class SubsetImportanceSampledKLLoss(BaseLoss):
    """
    Exact top-k head KL with an importance-sampled estimate of the teacher-tail KL.

    The efficient path is ``select_indices()`` followed by a model/lens forward
    on those indices and then ``forward_gathered()``. The gathered path requires
    the full-vocabulary student log normalizer so both head and tail use the
    true full-vocabulary ``log Q``.
    """

    def __init__(
        self,
        k_head: int = 256,
        k_tail: int = 256,
        tail_proposal: TailProposalType = "target",
        tail_proposal_alpha: float = 0.8,
        tail_proposal_tau: float = 0.7,
        reduction: ReductionType = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.k_head = k_head
        self.k_tail = k_tail
        self.tail_proposal = tail_proposal
        self.tail_proposal_alpha = tail_proposal_alpha
        self.tail_proposal_tau = tail_proposal_tau
        self._last_indices: Optional[torch.Tensor] = None
        self._last_p_head: Optional[torch.Tensor] = None
        self._last_tail_proposal_log_probs: Optional[torch.Tensor] = None

    def select_indices(
        self,
        teacher_logits: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top-k head plus sampled tail indices."""
        selected = select_head_tail_indices(
            teacher_logits,
            k_head=self.k_head,
            k_tail=self.k_tail,
            generator=generator,
            tail_proposal=self.tail_proposal,
            tail_proposal_alpha=self.tail_proposal_alpha,
            tail_proposal_tau=self.tail_proposal_tau,
            return_tail_proposal_log_probs=self.tail_proposal not in {"target", "teacher"},
        )
        if self.tail_proposal not in {"target", "teacher"}:
            indices, teacher_log_probs_selected, p_head, tail_proposal_log_probs = selected
        else:
            indices, teacher_log_probs_selected, p_head = selected
            tail_proposal_log_probs = None
        self._last_indices = indices
        self._last_p_head = p_head
        self._last_tail_proposal_log_probs = tail_proposal_log_probs
        return indices, teacher_log_probs_selected, p_head

    def forward_gathered(
        self,
        student_logits_selected: torch.Tensor,
        teacher_log_probs_selected: torch.Tensor,
        p_head: Optional[torch.Tensor] = None,
        student_log_normalizer: Optional[torch.Tensor] = None,
        tail_proposal_log_probs_selected: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute importance-sampled KL from pre-gathered selected logits."""
        return subset_is_kl_from_gathered(
            student_logits_selected,
            teacher_log_probs_selected,
            k_head=self.k_head,
            k_tail=self.k_tail,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
            tail_proposal_log_probs_selected=tail_proposal_log_probs_selected,
            attention_mask=attention_mask,
            reduction=self.reduction,
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute importance-sampled KL from full logits as a convenience path."""
        indices, teacher_log_probs_selected, p_head = self.select_indices(teacher_logits)
        student_selected = torch.gather(student_logits, -1, indices)
        student_log_normalizer = torch.logsumexp(student_logits.float(), dim=-1)
        return self.forward_gathered(
            student_selected,
            teacher_log_probs_selected,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
            tail_proposal_log_probs_selected=self._last_tail_proposal_log_probs,
            attention_mask=attention_mask,
        )

    @property
    def last_indices(self) -> Optional[torch.Tensor]:
        """Last selected indices."""
        return self._last_indices

    @property
    def last_p_head(self) -> Optional[torch.Tensor]:
        """Last teacher head mass."""
        return self._last_p_head

    @property
    def last_tail_proposal_log_probs(self) -> Optional[torch.Tensor]:
        """Last selected tail proposal log-probabilities."""
        return self._last_tail_proposal_log_probs

    def __repr__(self) -> str:
        return (
            f"SubsetImportanceSampledKLLoss(k_head={self.k_head}, k_tail={self.k_tail}, "
            f"tail_proposal={self.tail_proposal!r}, reduction={self.reduction!r})"
        )






