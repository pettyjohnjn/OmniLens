# src/subset_kl/sampling.py
"""
Sampling utilities for subset KL divergence estimation.

Implements Probability Proportional to Size (PPS) sampling and
unbiased subset-based estimation of full-vocabulary KL divergence.

For most use cases, the simple top-k approach via
`select_topk_indices()` is sufficient and more stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplingDiagnostics:
    """Diagnostics from PPS sampling for monitoring estimator quality."""
    num_unique_indices: int
    mean_inclusion_prob: float
    min_inclusion_prob: float
    max_importance_weight: float
    variance_proxy: Optional[float] = None
    
    def __repr__(self) -> str:
        return (
            f"SamplingDiagnostics(unique={self.num_unique_indices}, "
            f"mean_π={self.mean_inclusion_prob:.4f}, "
            f"min_π={self.min_inclusion_prob:.6f}, "
            f"max_w={self.max_importance_weight:.1f})"
        )


def pps_sample_indices_batched(
    log_probs: torch.Tensor,
    k_head: int,
    k_tail: int,
    oversample: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, SamplingDiagnostics]:
    """
    Probability Proportional to Size sampling with deterministic head.
    
    Combines top-k (deterministic head) with PPS sampling (stochastic tail)
    for a hybrid subset that captures both high-probability tokens and
    provides unbiased coverage of the tail distribution.
    
    Parameters
    ----------
    log_probs : torch.Tensor
        Log probabilities [N, V] where N = B*T (flattened positions).
    k_head : int
        Number of top tokens to include deterministically. Set to 0
        for pure importance sampling (no deterministic head).
    k_tail : int
        Number of additional tokens to sample from tail.
    oversample : int
        Oversample factor for PPS (draw oversample*k_tail then deduplicate).
        
    Returns
    -------
    indices : torch.Tensor
        Selected indices [N, S] where S <= k_head + k_tail.
    inclusion_probs : torch.Tensor
        Inclusion probabilities for each selected index [N, S].
    mask : torch.Tensor
        Valid mask for variable-length selections [N, S].
    diagnostics : SamplingDiagnostics
        Monitoring statistics.
    """
    N, V = log_probs.shape
    device = log_probs.device

    if k_head < 0 or k_tail < 0:
        raise ValueError("k_head and k_tail must be non-negative")
    if k_head + k_tail == 0:
        raise ValueError("k_head + k_tail must be > 0")
    
    # Ensure numerical stability
    log_probs = log_probs.float()
    probs = F.softmax(log_probs, dim=-1)
    
    # HEAD: Top-k deterministic indices
    if k_head > 0:
        _, top_idx = log_probs.topk(k_head, dim=-1)  # [N, k_head]
    else:
        top_idx = torch.empty(N, 0, device=device, dtype=torch.long)
    
    # Renormalize probabilities over tail
    tail_probs = probs.clone()
    if k_head > 0:
        tail_probs.scatter_(1, top_idx, 0.0)
        tail_sum = (1.0 - torch.gather(probs, 1, top_idx).sum(dim=-1, keepdim=True)).clamp_min(1e-12)
    else:
        tail_sum = tail_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    tail_probs = tail_probs / tail_sum
    
    # PPS sampling from tail with replacement
    num_draws = k_tail * oversample
    if num_draws > 0 and (tail_sum > 1e-10).any():
        # Sample indices proportional to tail_probs
        sampled = torch.multinomial(
            tail_probs,
            num_samples=min(num_draws, V - k_head),
            replacement=True
        )  # [N, num_draws]
        
        # Combine head and tail
        all_idx = torch.cat([top_idx, sampled], dim=-1)
        
        # Deduplicate via sort + unique detection
        all_idx_sorted, _ = all_idx.sort(dim=-1)
        diff = torch.diff(all_idx_sorted, dim=-1, prepend=all_idx_sorted[:, :1] - 1)
        keep_mask = diff != 0
        
        # Compact unique indices without a Python row loop
        counts = keep_mask.sum(dim=-1)
        S_max = min(k_head + k_tail, counts.max().item())
        positions = keep_mask.cumsum(dim=-1) - 1
        valid = keep_mask & (positions < S_max)

        indices = torch.zeros(N, S_max, device=device, dtype=torch.long)
        mask = torch.zeros(N, S_max, device=device, dtype=torch.bool)

        row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(all_idx_sorted)
        compact_rows = row_idx[valid]
        compact_cols = positions[valid]
        indices[compact_rows, compact_cols] = all_idx_sorted[valid]
        mask[compact_rows, compact_cols] = True
    else:
        # No tail sampling
        indices = top_idx
        mask = torch.ones(N, k_head, device=device, dtype=torch.bool)
    
    # Compute inclusion probabilities
    inclusion_probs = torch.ones_like(indices, dtype=torch.float32)
    
    if k_tail > 0 and num_draws > 0:
        # Gather probabilities for selected indices
        p_sel = torch.gather(probs, 1, indices)
        
        # Identify which are head (prob >= top-k threshold)
        if k_head > 0:
            p_head_min = torch.gather(probs, 1, top_idx[:, -1:])
            is_head = p_sel >= p_head_min
        else:
            is_head = torch.zeros_like(p_sel, dtype=torch.bool)
        
        # For tail items: π_i = 1 - (1-p_i)^m
        tail_inclusion = 1.0 - torch.pow(1.0 - p_sel, num_draws)
        inclusion_probs = torch.where(is_head, torch.ones_like(p_sel), tail_inclusion)
        inclusion_probs = inclusion_probs.clamp_min(1e-8)
    
    # Compute diagnostics
    valid_inclusion = inclusion_probs[mask]
    diagnostics = SamplingDiagnostics(
        num_unique_indices=int(mask.sum().item() / N) if N > 0 else 0,
        mean_inclusion_prob=valid_inclusion.mean().item() if valid_inclusion.numel() > 0 else 1.0,
        min_inclusion_prob=valid_inclusion.min().item() if valid_inclusion.numel() > 0 else 1.0,
        max_importance_weight=(1.0 / valid_inclusion.min()).item() if valid_inclusion.numel() > 0 else 1.0,
    )
    
    return indices, inclusion_probs, mask, diagnostics


