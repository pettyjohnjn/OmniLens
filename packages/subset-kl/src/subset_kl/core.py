# src/subset_kl/core.py
"""
Core functional interfaces for subset KL divergence.

This module provides the primary APIs for memory-efficient KL computation:

1. `select_topk_indices()` - Select which tokens to use
2. `subset_kl_from_gathered()` - Compute KL on pre-gathered subsets
3. `compute_subset_kl()` - All-in-one when you have full logits

The typical workflow for maximum efficiency:

    # 1. Select indices from teacher (subset-kl)
    indices, teacher_k = select_topk_indices(teacher_logits, k=256)
    
    # 2. Compute student logits ONLY for those indices (your model/lens)
    student_k = your_model.compute_logits_for_indices(hidden, indices)
    
    # 3. Compute KL on the subsets (subset-kl)
    loss = subset_kl_from_gathered(student_k, teacher_k)

This avoids materializing full [B, T, V] student logits.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import apply_reduction, ReductionType


TailProposalType = Literal["target", "teacher", "mixed", "tempered"]


# =============================================================================
# Index Selection
# =============================================================================

def select_topk_indices(
    teacher_logits: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k indices from the teacher distribution.
    
    This is the first step in memory-efficient subset KL. The returned
    indices should be passed to your model to compute student logits
    for ONLY these tokens.
    
    Parameters
    ----------
    teacher_logits : torch.Tensor
        Teacher logits [batch, seq, vocab] or [N, vocab].
    k : int
        Number of top tokens to select.
        
    Returns
    -------
    indices : torch.Tensor
        Top-k token indices [batch, seq, k] or [N, k].
    teacher_logits_k : torch.Tensor
        Teacher logits for selected indices [batch, seq, k] or [N, k].
        
    Examples
    --------
    >>> indices, teacher_k = select_topk_indices(teacher_logits, k=256)
    >>> student_k = model.forward_subset(hidden_states, indices)
    >>> loss = subset_kl_from_gathered(student_k, teacher_k)
    """
    # topk returns (values, indices)
    teacher_logits_k, indices = teacher_logits.topk(k=k, dim=-1)
    return indices, teacher_logits_k


def select_indices_with_sampling(
    teacher_logits: torch.Tensor,
    k_head: int,
    k_tail: int = 0,
    oversample: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select indices using top-k head + importance-sampled tail.
    
    For most cases, pure top-k (k_tail=0) is sufficient and more stable.
    Use this when you need an unbiased estimator.
    
    Parameters
    ----------
    teacher_logits : torch.Tensor
        Teacher logits [batch, seq, vocab].
    k_head : int
        Number of top tokens (deterministic).
    k_tail : int
        Number of additional sampled tokens (stochastic).
    oversample : int
        Oversample factor for PPS sampling.
        
    Returns
    -------
    indices : torch.Tensor
        Selected indices [batch, seq, S] where S <= k_head + k_tail.
    teacher_logits_selected : torch.Tensor
        Teacher logits for selected indices.
    inclusion_probs : torch.Tensor
        Inclusion probabilities for importance weighting.
    mask : torch.Tensor
        Valid mask for variable-length selections.
    """
    from .sampling import pps_sample_indices_batched
    
    B, T, V = teacher_logits.shape
    
    # Flatten for sampling
    teacher_flat = teacher_logits.view(B * T, V)
    log_probs = F.log_softmax(teacher_flat, dim=-1)
    
    indices, inc_probs, mask, _ = pps_sample_indices_batched(
        log_probs, k_head=k_head, k_tail=k_tail, oversample=oversample
    )
    
    # Gather teacher values
    teacher_selected = torch.gather(teacher_flat, -1, indices)
    
    # Reshape back to [B, T, S]
    S = indices.shape[-1]
    indices = indices.view(B, T, S)
    teacher_selected = teacher_selected.view(B, T, S)
    inc_probs = inc_probs.view(B, T, S)
    mask = mask.view(B, T, S)
    
    return indices, teacher_selected, inc_probs, mask


def select_indices_with_importance_sampling(
    teacher_logits: torch.Tensor,
    k: int,
    oversample: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select indices using pure importance sampling (no deterministic head).
    
    This is equivalent to select_indices_with_sampling(k_head=0, k_tail=k).
    
    Parameters
    ----------
    teacher_logits : torch.Tensor
        Teacher logits [batch, seq, vocab].
    k : int
        Number of sampled tokens (stochastic).
    oversample : int
        Oversample factor for PPS sampling.
        
    Returns
    -------
    indices : torch.Tensor
        Selected indices [batch, seq, S] where S <= k.
    teacher_logits_selected : torch.Tensor
        Teacher logits for selected indices.
    inclusion_probs : torch.Tensor
        Inclusion probabilities for importance weighting.
    mask : torch.Tensor
        Valid mask for variable-length selections.
    """
    return select_indices_with_sampling(
        teacher_logits=teacher_logits,
        k_head=0,
        k_tail=k,
        oversample=oversample,
    )


def _tail_proposal_from_probs(
    tail_probs: torch.Tensor,
    tail_mass: torch.Tensor,
    proposal: TailProposalType,
    alpha: float,
    tau: float,
) -> torch.Tensor:
    """Build a normalized proposal over non-head tail entries."""
    if proposal == "teacher":
        proposal = "target"
    if proposal not in {"target", "mixed", "tempered"}:
        raise ValueError(
            "tail_proposal must be one of: 'target', 'teacher', 'mixed', 'tempered'"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("tail_proposal_alpha must be in [0, 1]")
    if tau <= 0.0:
        raise ValueError("tail_proposal_tau must be positive")

    q_target = tail_probs / tail_mass
    if proposal == "target":
        return q_target

    q_explore = tail_probs.pow(tau)
    q_explore = q_explore / q_explore.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if proposal == "tempered":
        q = q_explore
    else:
        q = alpha * q_target + (1.0 - alpha) * q_explore
    return q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def select_head_tail_indices(
    teacher_logits: torch.Tensor,
    k_head: int,
    k_tail: int,
    generator: Optional[torch.Generator] = None,
    tail_proposal: TailProposalType = "target",
    tail_proposal_alpha: float = 0.8,
    tail_proposal_tau: float = 0.7,
    return_tail_proposal_log_probs: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select a deterministic top-k head and importance-sampled draws from the tail.

    Tail samples are drawn with replacement from the teacher distribution after
    removing the top-k head and renormalizing the remaining probability mass.
    The returned teacher values are full-vocabulary log-probabilities, not raw
    logits, so they can be used directly in the K2 tail term.

    Parameters
    ----------
    teacher_logits : torch.Tensor
        Teacher logits [..., vocab].
    k_head : int
        Number of top tokens to include deterministically.
    k_tail : int
        Number of tail tokens to sample from P(. | not in head).
    generator : Optional[torch.Generator]
        Optional torch random generator for reproducible sampling.
    tail_proposal : {"target", "teacher", "mixed", "tempered"}
        Proposal used for tail sampling. ``"target"``/``"teacher"`` preserves
        existing behavior by sampling from P(. | tail). ``"mixed"`` samples
        from alpha * P(. | tail) + (1 - alpha) * normalized(P_i ** tau).
        ``"tempered"`` uses only the tempered proposal.
    tail_proposal_alpha : float
        Mixture weight on the target tail proposal for ``tail_proposal="mixed"``.
    tail_proposal_tau : float
        Tempering exponent for the exploration proposal.
    return_tail_proposal_log_probs : bool
        If true, also return log proposal probabilities for sampled tail tokens.

    Returns
    -------
    indices : torch.Tensor
        Concatenated head and tail indices [..., k_head + k_tail].
    teacher_log_probs_selected : torch.Tensor
        Full-vocabulary teacher log-probabilities at selected indices.
    p_head : torch.Tensor
        Full-vocabulary teacher probability mass in the head [...].
    tail_proposal_log_probs_selected : torch.Tensor
        Returned only when ``return_tail_proposal_log_probs=True``. Contains
        proposal log-probabilities for selected tail entries, shape
        [..., k_tail].
    """
    if k_head < 0 or k_tail < 0:
        raise ValueError("k_head and k_tail must be non-negative")
    if k_head + k_tail == 0:
        raise ValueError("k_head + k_tail must be > 0")

    vocab = teacher_logits.shape[-1]
    if k_head > vocab:
        raise ValueError(f"k_head={k_head} exceeds vocab size {vocab}")
    if k_tail > 0 and k_head >= vocab:
        raise ValueError("Cannot sample tail tokens when k_head covers the vocabulary")

    leading_shape = teacher_logits.shape[:-1]
    teacher_flat = teacher_logits.reshape(-1, vocab)
    log_probs = F.log_softmax(teacher_flat.float(), dim=-1)
    probs = log_probs.exp()

    if k_head > 0:
        _, head_idx = teacher_flat.topk(k_head, dim=-1)
        head_probs = torch.gather(probs, -1, head_idx)
        p_head = head_probs.sum(dim=-1)
    else:
        head_idx = torch.empty(
            teacher_flat.shape[0], 0, device=teacher_logits.device, dtype=torch.long
        )
        p_head = torch.zeros(
            teacher_flat.shape[0], device=teacher_logits.device, dtype=probs.dtype
        )

    if k_tail > 0:
        tail_probs = probs.clone()
        if k_head > 0:
            tail_probs.scatter_(1, head_idx, 0.0)
        tail_mass = tail_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        proposal_probs = _tail_proposal_from_probs(
            tail_probs,
            tail_mass,
            proposal=tail_proposal,
            alpha=tail_proposal_alpha,
            tau=tail_proposal_tau,
        )
        tail_idx = torch.multinomial(
            proposal_probs,
            num_samples=k_tail,
            replacement=True,
            generator=generator,
        )
        tail_proposal_log_probs = torch.gather(
            proposal_probs.clamp_min(1e-45).log(), -1, tail_idx
        )
    else:
        tail_idx = torch.empty(
            teacher_flat.shape[0], 0, device=teacher_logits.device, dtype=torch.long
        )
        tail_proposal_log_probs = torch.empty(
            teacher_flat.shape[0],
            0,
            device=teacher_logits.device,
            dtype=log_probs.dtype,
        )

    indices_flat = torch.cat([head_idx, tail_idx], dim=-1)
    teacher_log_probs_selected = torch.gather(log_probs, -1, indices_flat)

    indices = indices_flat.reshape(*leading_shape, k_head + k_tail)
    teacher_log_probs_selected = teacher_log_probs_selected.reshape(
        *leading_shape, k_head + k_tail
    )
    p_head = p_head.reshape(*leading_shape)
    tail_proposal_log_probs = tail_proposal_log_probs.reshape(*leading_shape, k_tail)
    if return_tail_proposal_log_probs:
        return indices, teacher_log_probs_selected, p_head, tail_proposal_log_probs
    return indices, teacher_log_probs_selected, p_head


# =============================================================================
# KL Computation on Pre-Gathered Tensors
# =============================================================================

def subset_kl_from_gathered(
    student_logits_k: torch.Tensor,
    teacher_logits_k: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = "mean",
) -> torch.Tensor:
    """
    Compute KL divergence from pre-gathered subset logits.
    
    Both student and teacher logits should be for the SAME k indices
    (typically from `select_topk_indices`). Distributions are 
    renormalized over the k tokens before computing KL.

    Paper mapping: the Top-k truncation objective of Sec. 4.1
    (eq:KL_topk) -- both distributions restricted to the head set H and
    renormalized, which changes the objective (biased) but avoids the
    full-vocabulary projection.
    
    Parameters
    ----------
    student_logits_k : torch.Tensor
        Student logits for subset [batch, seq, k] or [N, k].
    teacher_logits_k : torch.Tensor
        Teacher logits for same subset [batch, seq, k] or [N, k].
    attention_mask : Optional[torch.Tensor]
        Mask [batch, seq] or [N]. Applied after KL computation.
    reduction : str
        "none", "mean", or "sum".
        
    Returns
    -------
    torch.Tensor
        KL divergence. Shape depends on reduction.
        
    Examples
    --------
    >>> indices, teacher_k = select_topk_indices(teacher_logits, k=256)
    >>> student_k = lens.forward(hidden, vocab_indices=indices).logits
    >>> loss = subset_kl_from_gathered(student_k, teacher_k)
    
    Notes
    -----
    The KL is computed as:
    
        KL(P || Q) = Σ_i P_i * (log P_i - log Q_i)
    
    where P and Q are renormalized over the k tokens. This is not
    the same as full-vocabulary KL, but for k >= 256 with typical
    LLM distributions, the difference is negligible (<1%).
    """
    # Renormalize both distributions over the subset
    teacher_logprobs = F.log_softmax(teacher_logits_k, dim=-1)
    student_logprobs = F.log_softmax(student_logits_k, dim=-1)
    
    # KL(teacher || student)
    teacher_probs = teacher_logprobs.exp()
    kl_per_token = (teacher_probs * (teacher_logprobs - student_logprobs)).sum(dim=-1)
    
    return apply_reduction(kl_per_token, attention_mask, reduction)




def subset_k2_kl_from_gathered(
    student_logits_selected: torch.Tensor,
    teacher_log_probs_selected: torch.Tensor,
    k_head: int,
    k_tail: int,
    p_head: Optional[torch.Tensor] = None,
    student_log_normalizer: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = "mean",
) -> torch.Tensor:
    """
    Compute top-k head KL plus an importance-sampled K2 penalty on sampled tail tokens.

    This implements the package form of

        KL_head + (1 - sg(P_head)) / k_tail * sum_i (log P(t_i) - log Q(t_i))^2

    where the first ``k_head`` entries are deterministic head tokens and the
    next ``k_tail`` entries are samples from the teacher tail. The head KL uses
    the same subset-renormalized top-k estimator as ``subset_kl_from_gathered``.
    For the tail K2 term, ``P`` and ``Q`` are full-vocabulary log-probabilities.
    Since selected student logits alone do not define full-vocabulary ``Q``,
    ``student_log_normalizer=logsumexp(student_logits_full, dim=-1)`` is
    required whenever ``k_tail > 0``.

    Parameters
    ----------
    student_logits_selected : torch.Tensor
        Student logits for selected indices [..., k_head + k_tail].
    teacher_log_probs_selected : torch.Tensor
        Full teacher log-probabilities for selected indices.
    k_head : int
        Number of deterministic head entries.
    k_tail : int
        Number of sampled tail entries.
    p_head : Optional[torch.Tensor]
        Teacher head mass [...]. If omitted, it is computed from the first
        ``k_head`` teacher log-probabilities.
    student_log_normalizer : Optional[torch.Tensor]
        Full-vocabulary student log normalizer with shape matching the leading
        dimensions of ``student_logits_selected``. Must remain differentiable
        for unbiased tail gradients.
    attention_mask : Optional[torch.Tensor]
        Mask matching the leading dimensions.
    reduction : str
        "none", "mean", or "sum".
    """
    if k_head < 0 or k_tail < 0:
        raise ValueError("k_head and k_tail must be non-negative")
    if k_head + k_tail == 0:
        raise ValueError("k_head + k_tail must be > 0")
    if student_logits_selected.shape != teacher_log_probs_selected.shape:
        raise ValueError("student and teacher selected tensors must have matching shapes")
    if student_logits_selected.shape[-1] != k_head + k_tail:
        raise ValueError("last dimension must equal k_head + k_tail")

    leading_shape = student_logits_selected.shape[:-1]
    per_position = torch.zeros(
        leading_shape,
        device=student_logits_selected.device,
        dtype=student_logits_selected.dtype,
    )

    if k_head > 0:
        student_head = student_logits_selected[..., :k_head]
        teacher_head = teacher_log_probs_selected[..., :k_head]
        teacher_head_log_probs = teacher_head - torch.logsumexp(
            teacher_head, dim=-1, keepdim=True
        )
        student_head_log_probs = F.log_softmax(student_head, dim=-1)
        teacher_head_probs = teacher_head_log_probs.exp()
        per_position = (
            teacher_head_probs * (teacher_head_log_probs - student_head_log_probs)
        ).sum(dim=-1)

    if k_tail > 0:
        if student_log_normalizer is None:
            raise ValueError(
                "student_log_normalizer is required when k_tail > 0 because "
                "the K2 tail term needs full-vocabulary student log-probabilities"
            )
        if student_log_normalizer.shape != leading_shape:
            raise ValueError("student_log_normalizer shape must match leading dimensions")

        if p_head is None:
            if k_head > 0:
                p_head = teacher_log_probs_selected[..., :k_head].exp().sum(dim=-1)
            else:
                p_head = torch.zeros(
                    leading_shape,
                    device=teacher_log_probs_selected.device,
                    dtype=teacher_log_probs_selected.dtype,
                )

        teacher_tail = teacher_log_probs_selected[..., k_head:]
        student_log_probs_selected = student_logits_selected - student_log_normalizer.unsqueeze(-1)
        student_tail = student_log_probs_selected[..., k_head:]
        k2_tail = (teacher_tail - student_tail).square().sum(dim=-1)
        per_position = per_position + (1.0 - p_head.detach()) * k2_tail / k_tail

    return apply_reduction(per_position, attention_mask, reduction)


def subset_k3_kl_from_gathered(
    student_logits_selected: torch.Tensor,
    teacher_log_probs_selected: torch.Tensor,
    k_head: int,
    k_tail: int,
    p_head: Optional[torch.Tensor] = None,
    student_log_normalizer: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = "mean",
) -> torch.Tensor:
    """
    Compute top-k head KL plus Schulman's K3 estimator on sampled tail tokens.

    The tail term uses Schulman's unbiased low-variance KL estimator

        exp(log Q(t_i) - log P(t_i)) - 1 - (log Q(t_i) - log P(t_i))

    for ``KL(P || Q)`` with samples ``t_i`` from the teacher tail ``P``. The
    first ``k_head`` entries are deterministic head tokens and the next
    ``k_tail`` entries are samples from the teacher tail. The head KL uses the
    same subset-renormalized top-k estimator as ``subset_kl_from_gathered``.
    For the tail K3 term, ``P`` and ``Q`` are full-vocabulary log-probabilities.
    Since selected student logits alone do not define full-vocabulary ``Q``,
    ``student_log_normalizer=logsumexp(student_logits_full, dim=-1)`` is
    required whenever ``k_tail > 0``.

    Parameters are identical to ``subset_k2_kl_from_gathered``.
    """
    if k_head < 0 or k_tail < 0:
        raise ValueError("k_head and k_tail must be non-negative")
    if k_head + k_tail == 0:
        raise ValueError("k_head + k_tail must be > 0")
    if student_logits_selected.shape != teacher_log_probs_selected.shape:
        raise ValueError("student and teacher selected tensors must have matching shapes")
    if student_logits_selected.shape[-1] != k_head + k_tail:
        raise ValueError("last dimension must equal k_head + k_tail")

    leading_shape = student_logits_selected.shape[:-1]
    per_position = torch.zeros(
        leading_shape,
        device=student_logits_selected.device,
        dtype=student_logits_selected.dtype,
    )

    if k_head > 0:
        student_head = student_logits_selected[..., :k_head]
        teacher_head = teacher_log_probs_selected[..., :k_head]
        teacher_head_log_probs = teacher_head - torch.logsumexp(
            teacher_head, dim=-1, keepdim=True
        )
        student_head_log_probs = F.log_softmax(student_head, dim=-1)
        teacher_head_probs = teacher_head_log_probs.exp()
        per_position = (
            teacher_head_probs * (teacher_head_log_probs - student_head_log_probs)
        ).sum(dim=-1)

    if k_tail > 0:
        if student_log_normalizer is None:
            raise ValueError(
                "student_log_normalizer is required when k_tail > 0 because "
                "the K3 tail term needs full-vocabulary student log-probabilities"
            )
        if student_log_normalizer.shape != leading_shape:
            raise ValueError("student_log_normalizer shape must match leading dimensions")

        if p_head is None:
            if k_head > 0:
                p_head = teacher_log_probs_selected[..., :k_head].exp().sum(dim=-1)
            else:
                p_head = torch.zeros(
                    leading_shape,
                    device=teacher_log_probs_selected.device,
                    dtype=teacher_log_probs_selected.dtype,
                )

        teacher_tail = teacher_log_probs_selected[..., k_head:]
        student_log_probs_selected = student_logits_selected - student_log_normalizer.unsqueeze(-1)
        student_tail = student_log_probs_selected[..., k_head:]
        log_ratio = student_tail - teacher_tail
        k3_tail = (torch.expm1(log_ratio) - log_ratio).sum(dim=-1)
        per_position = per_position + (1.0 - p_head.detach()) * k3_tail / k_tail

    return apply_reduction(per_position, attention_mask, reduction)


def subset_is_kl_from_gathered(
    student_logits_selected: torch.Tensor,
    teacher_log_probs_selected: torch.Tensor,
    k_head: int,
    k_tail: int,
    p_head: Optional[torch.Tensor] = None,
    student_log_normalizer: Optional[torch.Tensor] = None,
    tail_proposal_log_probs_selected: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = "mean",
    return_diagnostics: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute exact head KL plus an importance-sampled estimate of the teacher-tail KL.

    Paper mapping: the Top-k+IS estimator of Sec. 4.2 (eq:subset_KL) --
    exact head sum plus P/R-reweighted sampled tail; with a lens-independent
    positive tail proposal and the exact student log-partition, the estimate
    and its gradients are unbiased for the full KL (Theorem 1).

    The first ``k_head`` entries are deterministic head tokens. By default the
    next ``k_tail`` entries are samples from the teacher tail distribution. If
    ``tail_proposal_log_probs_selected`` is supplied, the tail uses unbiased
    importance sampling with absolute weights ``P_i / q_i``.

        sum_head P_i (log P_i - log Q_i)
        + (1 - P_head) / k_tail * sum_tail_samples (log P_i - log Q_i)

    The tail term needs full-vocabulary student log-probabilities, so
    ``student_log_normalizer=logsumexp(student_logits_full, dim=-1)`` is
    required whenever ``k_tail > 0``.
    """
    if k_head < 0 or k_tail < 0:
        raise ValueError("k_head and k_tail must be non-negative")
    if k_head + k_tail == 0:
        raise ValueError("k_head + k_tail must be > 0")
    if student_logits_selected.shape != teacher_log_probs_selected.shape:
        raise ValueError("student and teacher selected tensors must have matching shapes")
    if student_logits_selected.shape[-1] != k_head + k_tail:
        raise ValueError("last dimension must equal k_head + k_tail")

    leading_shape = student_logits_selected.shape[:-1]
    if student_log_normalizer is None:
        raise ValueError(
            "student_log_normalizer is required because the importance-sampled estimator "
            "uses full-vocabulary student log-probabilities"
        )
    if student_log_normalizer.shape != leading_shape:
        raise ValueError("student_log_normalizer shape must match leading dimensions")

    per_position = torch.zeros(
        leading_shape,
        device=student_logits_selected.device,
        dtype=student_logits_selected.dtype,
    )

    if k_head > 0:
        student_head_log_probs = (
            student_logits_selected[..., :k_head]
            - student_log_normalizer.unsqueeze(-1)
        )
        teacher_head = teacher_log_probs_selected[..., :k_head]
        per_position = (
            teacher_head.exp() * (teacher_head - student_head_log_probs)
        ).sum(dim=-1)

    if k_tail > 0:
        if p_head is None:
            if k_head > 0:
                p_head = teacher_log_probs_selected[..., :k_head].exp().sum(dim=-1)
            else:
                p_head = torch.zeros(
                    leading_shape,
                    device=teacher_log_probs_selected.device,
                    dtype=teacher_log_probs_selected.dtype,
                )

        teacher_tail = teacher_log_probs_selected[..., k_head:]
        student_log_probs_selected = student_logits_selected - student_log_normalizer.unsqueeze(-1)
        student_tail = student_log_probs_selected[..., k_head:]
        tail_terms = teacher_tail - student_tail
        if tail_proposal_log_probs_selected is None:
            tail_estimate = (1.0 - p_head.detach()) * tail_terms.sum(dim=-1) / k_tail
            weights = None
        else:
            if tail_proposal_log_probs_selected.shape != teacher_tail.shape:
                raise ValueError(
                    "tail_proposal_log_probs_selected shape must match tail selected shape"
                )
            weights = (teacher_tail - tail_proposal_log_probs_selected).exp()
            tail_estimate = (weights * tail_terms).sum(dim=-1) / k_tail
        per_position = per_position + tail_estimate

    loss = apply_reduction(per_position, attention_mask, reduction)
    if return_diagnostics:
        diagnostics = _tail_estimator_diagnostics(
            weights=weights if k_tail > 0 and tail_proposal_log_probs_selected is not None else None,
            tail_estimate=tail_estimate if k_tail > 0 else None,
            total_estimate=per_position,
            attention_mask=attention_mask,
        )
        return loss, diagnostics
    return loss


def _tail_estimator_diagnostics(
    weights: Optional[torch.Tensor],
    tail_estimate: Optional[torch.Tensor],
    total_estimate: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Dict[str, float]:
    """Summarize per-position tail estimates and sampled IS weights."""
    if attention_mask is not None:
        position_mask = attention_mask.to(dtype=torch.bool)
        estimates = total_estimate[position_mask]
        tails = tail_estimate[position_mask] if tail_estimate is not None else None
        ws = weights[position_mask] if weights is not None else None
    else:
        estimates = total_estimate.reshape(-1)
        tails = tail_estimate.reshape(-1) if tail_estimate is not None else None
        ws = weights.reshape(-1, weights.shape[-1]) if weights is not None else None

    result: Dict[str, float] = {
        "total_estimate_mean": estimates.float().mean().item() if estimates.numel() else 0.0,
    }
    if tails is not None and tails.numel():
        result["tail_contribution_mean"] = tails.float().mean().item()
    if ws is not None and ws.numel():
        ws_f = ws.float()
        ess = ws_f.sum(dim=-1).square() / ws_f.square().sum(dim=-1).clamp_min(1e-12)
        result.update(
            {
                "tail_ess_mean": ess.mean().item(),
                "tail_ess_min": ess.min().item(),
                "tail_ess_max": ess.max().item(),
                "tail_max_weight": ws_f.max().item(),
                "tail_weight_variance": ws_f.var(unbiased=False).item(),
                "tail_extreme_weight_frequency": (
                    ws_f > (10.0 * ws_f.mean().clamp_min(1e-12))
                ).float().mean().item(),
            }
        )
    return result




# =============================================================================
# Convenience: All-in-one when you have full logits
# =============================================================================

def compute_subset_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    k: int = 256,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = "mean",
) -> torch.Tensor:
    """
    Compute subset KL when you have full logits (convenience function).
    
    NOTE: This function still requires full [B, T, V] student logits,
    so it doesn't provide memory savings over full KL. Use this for:
    - Testing/validation
    - When student logits are already computed
    - Comparing subset vs full KL
    
    For memory efficiency, use `select_topk_indices()` + your model's
    subset forward + `subset_kl_from_gathered()`.
    
    Parameters
    ----------
    student_logits : torch.Tensor
        Student logits [batch, seq, vocab].
    teacher_logits : torch.Tensor
        Teacher logits [batch, seq, vocab].
    k : int
        Number of top tokens.
    attention_mask : Optional[torch.Tensor]
        Mask [batch, seq].
    reduction : str
        "none", "mean", or "sum".
        
    Returns
    -------
    torch.Tensor
        Subset KL divergence.
    """
    # Select top-k from teacher
    indices, teacher_k = select_topk_indices(teacher_logits, k)
    
    # Gather student logits (this is why we need full student logits)
    student_k = torch.gather(student_logits, -1, indices)
    
    # Compute KL
    return subset_kl_from_gathered(student_k, teacher_k, attention_mask, reduction)


def compute_subset_k2_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    k_head: int = 256,
    k_tail: int = 256,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = "mean",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Compute the head top-k plus tail K2 subset KL when full logits are available.

    This is a convenience function for testing and validation. For memory
    efficiency, call ``select_head_tail_indices()``, compute student logits only
    for the returned indices, then call ``subset_k2_kl_from_gathered()``.
    """
    indices, teacher_log_probs_selected, p_head = select_head_tail_indices(
        teacher_logits, k_head=k_head, k_tail=k_tail, generator=generator
    )
    student_selected = torch.gather(student_logits, -1, indices)
    student_log_normalizer = torch.logsumexp(student_logits.float(), dim=-1)
    return subset_k2_kl_from_gathered(
        student_selected,
        teacher_log_probs_selected,
        k_head=k_head,
        k_tail=k_tail,
        p_head=p_head,
        student_log_normalizer=student_log_normalizer,
        attention_mask=attention_mask,
        reduction=reduction,
    )


def compute_subset_k3_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    k_head: int = 256,
    k_tail: int = 256,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = "mean",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Compute the head top-k plus tail K3 subset KL when full logits are available.

    This is a convenience function for testing and validation. For memory
    efficiency, call ``select_head_tail_indices()``, compute student logits only
    for the returned indices, then call ``subset_k3_kl_from_gathered()``.
    """
    indices, teacher_log_probs_selected, p_head = select_head_tail_indices(
        teacher_logits, k_head=k_head, k_tail=k_tail, generator=generator
    )
    student_selected = torch.gather(student_logits, -1, indices)
    student_log_normalizer = torch.logsumexp(student_logits.float(), dim=-1)
    return subset_k3_kl_from_gathered(
        student_selected,
        teacher_log_probs_selected,
        k_head=k_head,
        k_tail=k_tail,
        p_head=p_head,
        student_log_normalizer=student_log_normalizer,
        attention_mask=attention_mask,
        reduction=reduction,
    )


def compute_subset_is_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    k_head: int = 256,
    k_tail: int = 256,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = "mean",
    generator: Optional[torch.Generator] = None,
    tail_proposal: TailProposalType = "target",
    tail_proposal_alpha: float = 0.8,
    tail_proposal_tau: float = 0.7,
    return_diagnostics: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute exact head plus importance-sampled teacher-tail KL when full logits are available.

    This is a convenience function for testing and validation. For memory
    efficiency, call ``select_head_tail_indices()``, compute selected student
    logits and the full-vocabulary student log normalizer, then call
    ``subset_is_kl_from_gathered()``.
    """
    use_explicit_proposal = tail_proposal not in {"target", "teacher"}
    selected = select_head_tail_indices(
        teacher_logits,
        k_head=k_head,
        k_tail=k_tail,
        generator=generator,
        tail_proposal=tail_proposal,
        tail_proposal_alpha=tail_proposal_alpha,
        tail_proposal_tau=tail_proposal_tau,
        return_tail_proposal_log_probs=use_explicit_proposal,
    )
    if use_explicit_proposal:
        indices, teacher_log_probs_selected, p_head, tail_proposal_log_probs = selected
    else:
        indices, teacher_log_probs_selected, p_head = selected
        tail_proposal_log_probs = None
    student_selected = torch.gather(student_logits, -1, indices)
    student_log_normalizer = torch.logsumexp(student_logits.float(), dim=-1)
    return subset_is_kl_from_gathered(
        student_selected,
        teacher_log_probs_selected,
        k_head=k_head,
        k_tail=k_tail,
        p_head=p_head,
        student_log_normalizer=student_log_normalizer,
        tail_proposal_log_probs_selected=tail_proposal_log_probs,
        attention_mask=attention_mask,
        reduction=reduction,
        return_diagnostics=return_diagnostics,
    )




# =============================================================================
# Full KL for comparison
# =============================================================================

def full_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = "mean",
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute full-vocabulary KL divergence (baseline).
    
    This is O(B*T*V) memory - use subset_kl for large vocabularies.
    
    Parameters
    ----------
    student_logits : torch.Tensor
        Student logits [batch, seq, vocab].
    teacher_logits : torch.Tensor
        Teacher logits [batch, seq, vocab].
    attention_mask : Optional[torch.Tensor]
        Mask [batch, seq].
    reduction : str
        "none", "mean", or "sum".
    temperature : float
        Temperature scaling (1.0 = no scaling).
    Returns
    -------
    torch.Tensor
        Full KL divergence.
    """
    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
    
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
    student_logprobs = F.log_softmax(student_logits, dim=-1)
    
    teacher_probs = teacher_logprobs.exp()
    kl_per_token = (teacher_probs * (teacher_logprobs - student_logprobs)).sum(dim=-1)
    
    return apply_reduction(kl_per_token, attention_mask, reduction)
