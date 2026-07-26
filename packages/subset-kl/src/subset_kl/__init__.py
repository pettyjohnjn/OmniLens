# src/subset_kl/__init__.py
"""
subset-kl: Memory-efficient KL divergence for large vocabularies.

This package provides memory-efficient KL divergence for large-vocabulary
models. Instead of computing full O(B*T*V) KL, it approximates using top-k
tokens, reducing memory to O(B*T*k) where k << V.

Architecture
------------
This package provides KL math only. It does not handle:
- Model weights or architectures
- Computing student logits from hidden states
- CUDA kernels for efficient matmul (see indexed_logits for that)

Your model/lens is responsible for:
1. Getting hidden states from the base model
2. Computing student logits for the selected indices

Workflow for Maximum Efficiency
-------------------------------
>>> from subset_kl import select_topk_indices, subset_kl_from_gathered
>>> 
>>> # Step 1: Select indices from teacher (this package)
>>> indices, teacher_k = select_topk_indices(teacher_logits, k=256)
>>> 
>>> # Step 2: Compute student logits ONLY for those indices (YOUR model)
>>> # This is where you use indexed_logits or your lens's vocab_indices param
>>> student_k = your_lens.forward(hidden, vocab_indices=indices).logits
>>> 
>>> # Step 3: Compute KL on the subsets (this package)
>>> loss = subset_kl_from_gathered(student_k, teacher_k, attention_mask)

This avoids materializing full [B, T, V] student logits.

Quick Start (when you have full logits)
---------------------------------------
>>> from subset_kl import SubsetKLLoss
>>> 
>>> loss_fn = SubsetKLLoss(k=256)
>>> loss = loss_fn(student_logits, teacher_logits)  # Requires full logits

Or functional:

>>> from subset_kl import compute_subset_kl
>>> loss = compute_subset_kl(student_logits, teacher_logits, k=256)

Memory Comparison
-----------------
For V=128k vocabulary, B=2, T=1024:

- Full KL: [2, 1024, 128000] = 1 GB (fp32)
- Subset KL (k=256): [2, 1024, 256] = 2 MB (500x reduction)
"""

__version__ = "0.1.0"

# =============================================================================
# Core Functional Interface (RECOMMENDED)
# =============================================================================
from .core import (
    TailProposalType,
    # Index selection
    select_topk_indices,
    select_head_tail_indices,
    select_indices_with_sampling,
    select_indices_with_importance_sampling,
    # KL computation on pre-gathered tensors
    subset_kl_from_gathered,
    subset_k2_kl_from_gathered,
    subset_k3_kl_from_gathered,
    subset_is_kl_from_gathered,
    # Convenience (when you have full logits)
    compute_subset_kl,
    compute_subset_k2_kl,
    compute_subset_k3_kl,
    compute_subset_is_kl,
    full_kl,
)

# =============================================================================
# Class-based Interface
# =============================================================================
from .losses import (
    SubsetKLLoss,
    SubsetK2KLLoss,
    SubsetK3KLLoss,
    SubsetImportanceSampledKLLoss,
    KLDivergenceLoss,
)

# =============================================================================
# Base Classes and Types
# =============================================================================
from .base import (
    BaseLoss,
    ReductionType,
    apply_reduction,
)

# =============================================================================
# Sampling Utilities (Advanced)
# =============================================================================
from .sampling import (
    pps_sample_indices_batched,
    SamplingDiagnostics,
)

__all__ = [
    # Version
    "__version__",
    # Core functional (recommended)
    "select_topk_indices",
    "TailProposalType",
    "select_head_tail_indices",
    "select_indices_with_sampling",
    "select_indices_with_importance_sampling",
    "subset_kl_from_gathered",
    "subset_k2_kl_from_gathered",
    "subset_k3_kl_from_gathered",
    "subset_is_kl_from_gathered",
    "compute_subset_kl",
    "compute_subset_k2_kl",
    "compute_subset_k3_kl",
    "compute_subset_is_kl",
    "full_kl",
    # Class-based
    "SubsetKLLoss",
    "SubsetK2KLLoss",
    "SubsetK3KLLoss",
    "SubsetImportanceSampledKLLoss",
    "KLDivergenceLoss",
    # Base
    "BaseLoss",
    "ReductionType",
    "apply_reduction",
    # Sampling
    "pps_sample_indices_batched",
    "SamplingDiagnostics",
]
