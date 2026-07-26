# tests/test_subset_kl.py
"""Tests for subset-kl package."""

import pytest
import torch
import torch.nn.functional as F

from subset_kl import (
    # Core functional
    select_topk_indices,
    select_head_tail_indices,
    select_indices_with_sampling,
    select_indices_with_importance_sampling,
    subset_kl_from_gathered,
    subset_k2_kl_from_gathered,
    subset_k3_kl_from_gathered,
    subset_is_kl_from_gathered,
    compute_subset_kl,
    compute_subset_k2_kl,
    compute_subset_k3_kl,
    compute_subset_is_kl,
    full_kl,
    # Classes
    SubsetKLLoss,
    SubsetK2KLLoss,
    SubsetK3KLLoss,
    SubsetImportanceSampledKLLoss,
    KLDivergenceLoss,
    # Sampling
    pps_sample_indices_batched,
    # Base
    apply_reduction,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def logits():
    """Standard test logits."""
    torch.manual_seed(42)
    B, T, V = 2, 8, 1000
    teacher = torch.randn(B, T, V) * 2  # Peaked distribution
    student = torch.randn(B, T, V, requires_grad=True)
    return teacher, student


@pytest.fixture
def small_logits():
    """Small logits for quick tests."""
    torch.manual_seed(42)
    B, T, V = 1, 4, 100
    teacher = torch.randn(B, T, V)
    student = torch.randn(B, T, V, requires_grad=True)
    return teacher, student


@pytest.fixture
def attention_mask():
    """Standard attention mask."""
    B, T = 2, 8
    mask = torch.ones(B, T)
    mask[0, -2:] = 0
    return mask


# =============================================================================
# Core Functional Tests
# =============================================================================

class TestSelectTopkIndices:
    """Tests for select_topk_indices."""
    
    def test_basic_shapes(self, small_logits):
        """Test output shapes."""
        teacher, _ = small_logits
        B, T, V = teacher.shape
        k = 32
        
        indices, teacher_k = select_topk_indices(teacher, k)
        
        assert indices.shape == (B, T, k)
        assert teacher_k.shape == (B, T, k)
    
    def test_indices_are_valid(self, small_logits):
        """Test indices are in valid range."""
        teacher, _ = small_logits
        V = teacher.shape[-1]
        
        indices, _ = select_topk_indices(teacher, k=32)
        
        assert (indices >= 0).all()
        assert (indices < V).all()
    
    def test_indices_are_top_k(self, small_logits):
        """Test that indices correspond to highest logits."""
        teacher, _ = small_logits
        k = 10
        
        indices, teacher_k = select_topk_indices(teacher, k)
        
        # Gather should give same values
        gathered = torch.gather(teacher, -1, indices)
        assert torch.allclose(gathered, teacher_k)


class TestSelectIndicesWithImportanceSampling:
    """Tests for select_indices_with_importance_sampling."""
    
    def test_basic_shapes(self, small_logits):
        """Test output shapes."""
        teacher, _ = small_logits
        B, T, V = teacher.shape
        k = 16
        
        indices, teacher_k, inc_probs, mask = select_indices_with_importance_sampling(
            teacher, k=k, oversample=10
        )
        S = indices.shape[-1]
        
        assert S <= k
        assert indices.shape == (B, T, S)
        assert teacher_k.shape == (B, T, S)
        assert inc_probs.shape == (B, T, S)
        assert mask.shape == (B, T, S)
        assert mask.any()


class TestSelectHeadTailIndices:
    """Tests for top-k head plus teacher-tail sampling."""

    def test_basic_shapes(self, small_logits):
        """Test output shapes."""
        teacher, _ = small_logits
        B, T, V = teacher.shape

        indices, teacher_log_probs, p_head = select_head_tail_indices(
            teacher, k_head=12, k_tail=7
        )

        assert indices.shape == (B, T, 19)
        assert teacher_log_probs.shape == (B, T, 19)
        assert p_head.shape == (B, T)
        assert (indices >= 0).all()
        assert (indices < V).all()
        assert (p_head >= 0).all()
        assert (p_head <= 1).all()

    def test_tail_excludes_head(self, small_logits):
        """Test sampled tail tokens do not include deterministic head tokens."""
        teacher, _ = small_logits
        k_head = 10
        k_tail = 16

        indices, _, _ = select_head_tail_indices(teacher, k_head=k_head, k_tail=k_tail)
        head = indices[..., :k_head]
        tail = indices[..., k_head:]

        assert not (tail.unsqueeze(-1) == head.unsqueeze(-2)).any()

    def test_mixed_tail_proposal_returns_selected_log_probs(self, small_logits):
        """Test mixed proposal samples from the tail and returns q for weights."""
        teacher, _ = small_logits
        k_head = 10
        k_tail = 16

        indices, teacher_log_probs, p_head, proposal_log_probs = select_head_tail_indices(
            teacher,
            k_head=k_head,
            k_tail=k_tail,
            tail_proposal="mixed",
            tail_proposal_alpha=0.8,
            tail_proposal_tau=0.7,
            return_tail_proposal_log_probs=True,
        )

        assert indices.shape[-1] == k_head + k_tail
        assert teacher_log_probs.shape == indices.shape
        assert p_head.shape == teacher.shape[:-1]
        assert proposal_log_probs.shape == teacher_log_probs[..., k_head:].shape
        assert torch.isfinite(proposal_log_probs).all()
        assert not (
            indices[..., k_head:].unsqueeze(-1) == indices[..., :k_head].unsqueeze(-2)
        ).any()


class TestSubsetKLFromGathered:
    """Tests for subset_kl_from_gathered."""
    
    def test_basic(self, small_logits):
        """Test basic computation."""
        teacher, student = small_logits
        k = 32
        
        indices, teacher_k = select_topk_indices(teacher, k)
        student_k = torch.gather(student, -1, indices)
        
        loss = subset_kl_from_gathered(student_k, teacher_k)
        
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # KL is non-negative
        assert torch.isfinite(loss)
    
    def test_identical_distributions(self):
        """Test KL=0 for identical distributions."""
        B, T, k = 1, 4, 32
        logits_k = torch.randn(B, T, k)
        
        loss = subset_kl_from_gathered(logits_k, logits_k)
        
        assert loss.item() < 1e-5
    
    def test_gradient_flow(self, small_logits):
        """Test gradients flow correctly."""
        teacher, student = small_logits
        k = 32
        
        indices, teacher_k = select_topk_indices(teacher, k)
        student_k = torch.gather(student, -1, indices)
        
        loss = subset_kl_from_gathered(student_k, teacher_k)
        loss.backward()
        
        assert student.grad is not None
        assert torch.isfinite(student.grad).all()
    
    def test_reduction_none(self, small_logits):
        """Test reduction='none' returns per-token."""
        teacher, student = small_logits
        B, T, V = teacher.shape
        k = 32
        
        indices, teacher_k = select_topk_indices(teacher, k)
        student_k = torch.gather(student, -1, indices)
        
        loss = subset_kl_from_gathered(student_k, teacher_k, reduction="none")
        
        assert loss.shape == (B, T)
    
    def test_with_attention_mask(self, small_logits):
        """Test attention mask is applied."""
        teacher, student = small_logits
        B, T, V = teacher.shape
        k = 32
        
        mask = torch.ones(B, T)
        mask[0, -1] = 0
        
        indices, teacher_k = select_topk_indices(teacher, k)
        student_k = torch.gather(student, -1, indices)
        
        loss_masked = subset_kl_from_gathered(student_k, teacher_k, mask)
        loss_unmasked = subset_kl_from_gathered(student_k, teacher_k)
        
        # Should be different (masked excludes some positions)
        assert not torch.allclose(loss_masked, loss_unmasked)


class TestSubsetK2KLFromGathered:
    """Tests for the subset K2 tail objective."""

    def test_basic(self, small_logits):
        """Test basic computation."""
        teacher, student = small_logits
        k_head, k_tail = 16, 8

        indices, teacher_log_probs, p_head = select_head_tail_indices(
            teacher, k_head=k_head, k_tail=k_tail
        )
        student_selected = torch.gather(student, -1, indices)
        student_log_normalizer = torch.logsumexp(student.float(), dim=-1)

        loss = subset_k2_kl_from_gathered(
            student_selected,
            teacher_log_probs,
            k_head=k_head,
            k_tail=k_tail,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
        )

        assert loss.dim() == 0
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_gradient_flow(self, small_logits):
        """Test gradients flow through selected student logits."""
        teacher, student = small_logits
        k_head, k_tail = 16, 8

        indices, teacher_log_probs, p_head = select_head_tail_indices(
            teacher, k_head=k_head, k_tail=k_tail
        )
        student_selected = torch.gather(student, -1, indices)
        student_log_normalizer = torch.logsumexp(student.float(), dim=-1)
        loss = subset_k2_kl_from_gathered(
            student_selected,
            teacher_log_probs,
            k_head=k_head,
            k_tail=k_tail,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
        )
        loss.backward()

        assert student.grad is not None
        assert torch.isfinite(student.grad).all()

    def test_requires_full_student_normalizer_for_tail(self, small_logits):
        """Test the K2 tail cannot silently use selected-subset Q."""
        teacher, student = small_logits
        k_head, k_tail = 16, 8

        indices, teacher_log_probs, p_head = select_head_tail_indices(
            teacher, k_head=k_head, k_tail=k_tail
        )
        student_selected = torch.gather(student, -1, indices)

        with pytest.raises(ValueError, match="student_log_normalizer is required"):
            subset_k2_kl_from_gathered(
                student_selected,
                teacher_log_probs,
                k_head=k_head,
                k_tail=k_tail,
                p_head=p_head,
            )

    def test_tail_matches_full_vocab_k2_formula(self, small_logits):
        """Test tail uses log Q from the full student distribution."""
        teacher, student = small_logits
        k_head, k_tail = 16, 8
        generator = torch.Generator().manual_seed(123)

        indices, teacher_log_probs, p_head = select_head_tail_indices(
            teacher, k_head=k_head, k_tail=k_tail, generator=generator
        )
        student_selected = torch.gather(student, -1, indices)
        student_log_normalizer = torch.logsumexp(student.float(), dim=-1)

        actual = subset_k2_kl_from_gathered(
            student_selected,
            teacher_log_probs,
            k_head=k_head,
            k_tail=k_tail,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
            reduction="none",
        )

        head_idx = indices[..., :k_head]
        tail_idx = indices[..., k_head:]
        teacher_head = torch.gather(teacher, -1, head_idx)
        student_head = torch.gather(student, -1, head_idx)
        teacher_head_log_probs = F.log_softmax(teacher_head, dim=-1)
        student_head_log_probs = F.log_softmax(student_head, dim=-1)
        head = (
            teacher_head_log_probs.exp()
            * (teacher_head_log_probs - student_head_log_probs)
        ).sum(dim=-1)

        teacher_full_log_probs = F.log_softmax(teacher.float(), dim=-1)
        student_full_log_probs = F.log_softmax(student.float(), dim=-1)
        teacher_tail = torch.gather(teacher_full_log_probs, -1, tail_idx)
        student_tail = torch.gather(student_full_log_probs, -1, tail_idx)
        expected = head + (1.0 - p_head.detach()) * (
            teacher_tail - student_tail
        ).square().sum(dim=-1) / k_tail

        assert torch.allclose(actual, expected)

    def test_no_tail_matches_topk(self, small_logits):
        """Test k_tail=0 matches the existing top-k objective."""
        teacher, student = small_logits
        k_head = 16

        indices, teacher_log_probs, p_head = select_head_tail_indices(
            teacher, k_head=k_head, k_tail=0
        )
        student_selected = torch.gather(student.detach(), -1, indices)

        actual = subset_k2_kl_from_gathered(
            student_selected,
            teacher_log_probs,
            k_head=k_head,
            k_tail=0,
            p_head=p_head,
        )
        expected = subset_kl_from_gathered(student_selected, teacher_log_probs)

        assert torch.allclose(actual, expected)


class TestComputeSubsetKL:
    """Tests for compute_subset_kl convenience function."""
    
    def test_matches_manual(self, small_logits):
        """Test it matches manual computation."""
        teacher, student = small_logits
        k = 32
        
        # Manual
        indices, teacher_k = select_topk_indices(teacher, k)
        student_k = torch.gather(student.detach(), -1, indices)
        expected = subset_kl_from_gathered(student_k, teacher_k)
        
        # Convenience function
        actual = compute_subset_kl(student.detach(), teacher, k=k)
        
        assert torch.allclose(expected, actual)


class TestComputeSubsetK2KL:
    """Tests for compute_subset_k2_kl convenience function."""

    def test_matches_manual_with_seeded_generator(self, small_logits):
        """Test convenience path matches manual selection with the same seed."""
        teacher, student = small_logits
        k_head, k_tail = 16, 8

        manual_generator = torch.Generator().manual_seed(123)
        indices, teacher_log_probs, p_head = select_head_tail_indices(
            teacher,
            k_head=k_head,
            k_tail=k_tail,
            generator=manual_generator,
        )
        student_selected = torch.gather(student.detach(), -1, indices)
        student_log_normalizer = torch.logsumexp(student.detach().float(), dim=-1)
        expected = subset_k2_kl_from_gathered(
            student_selected,
            teacher_log_probs,
            k_head=k_head,
            k_tail=k_tail,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
        )

        actual_generator = torch.Generator().manual_seed(123)
        actual = compute_subset_k2_kl(
            student.detach(),
            teacher,
            k_head=k_head,
            k_tail=k_tail,
            generator=actual_generator,
        )

        assert torch.allclose(expected, actual)


class TestSubsetK3KLFromGathered:
    """Tests for the subset K3 tail objective."""

    def test_basic(self, small_logits):
        """Test basic computation."""
        teacher, student = small_logits
        k_head, k_tail = 16, 8

        indices, teacher_log_probs, p_head = select_head_tail_indices(
            teacher, k_head=k_head, k_tail=k_tail
        )
        student_selected = torch.gather(student, -1, indices)
        student_log_normalizer = torch.logsumexp(student.float(), dim=-1)

        loss = subset_k3_kl_from_gathered(
            student_selected,
            teacher_log_probs,
            k_head=k_head,
            k_tail=k_tail,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
        )

        assert loss.dim() == 0
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_gradient_flow(self, small_logits):
        """Test gradients flow through selected student logits."""
        teacher, student = small_logits
        k_head, k_tail = 16, 8

        indices, teacher_log_probs, p_head = select_head_tail_indices(
            teacher, k_head=k_head, k_tail=k_tail
        )
        student_selected = torch.gather(student, -1, indices)
        student_log_normalizer = torch.logsumexp(student.float(), dim=-1)
        loss = subset_k3_kl_from_gathered(
            student_selected,
            teacher_log_probs,
            k_head=k_head,
            k_tail=k_tail,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
        )
        loss.backward()

        assert student.grad is not None
        assert torch.isfinite(student.grad).all()

    def test_requires_full_student_normalizer_for_tail(self, small_logits):
        """Test the K3 tail cannot silently use selected-subset Q."""
        teacher, student = small_logits
        k_head, k_tail = 16, 8

        indices, teacher_log_probs, p_head = select_head_tail_indices(
            teacher, k_head=k_head, k_tail=k_tail
        )
        student_selected = torch.gather(student, -1, indices)

        with pytest.raises(ValueError, match="student_log_normalizer is required"):
            subset_k3_kl_from_gathered(
                student_selected,
                teacher_log_probs,
                k_head=k_head,
                k_tail=k_tail,
                p_head=p_head,
            )

    def test_tail_matches_full_vocab_k3_formula(self, small_logits):
        """Test tail uses Schulman's K3 formula with full log Q."""
        teacher, student = small_logits
        k_head, k_tail = 16, 8
        generator = torch.Generator().manual_seed(123)

        indices, teacher_log_probs, p_head = select_head_tail_indices(
            teacher, k_head=k_head, k_tail=k_tail, generator=generator
        )
        student_selected = torch.gather(student, -1, indices)
        student_log_normalizer = torch.logsumexp(student.float(), dim=-1)

        actual = subset_k3_kl_from_gathered(
            student_selected,
            teacher_log_probs,
            k_head=k_head,
            k_tail=k_tail,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
            reduction="none",
        )

        head_idx = indices[..., :k_head]
        tail_idx = indices[..., k_head:]
        teacher_head = torch.gather(teacher, -1, head_idx)
        student_head = torch.gather(student, -1, head_idx)
        teacher_head_log_probs = F.log_softmax(teacher_head, dim=-1)
        student_head_log_probs = F.log_softmax(student_head, dim=-1)
        head = (
            teacher_head_log_probs.exp()
            * (teacher_head_log_probs - student_head_log_probs)
        ).sum(dim=-1)

        teacher_full_log_probs = F.log_softmax(teacher.float(), dim=-1)
        student_full_log_probs = F.log_softmax(student.float(), dim=-1)
        teacher_tail = torch.gather(teacher_full_log_probs, -1, tail_idx)
        student_tail = torch.gather(student_full_log_probs, -1, tail_idx)
        log_ratio = student_tail - teacher_tail
        expected = head + (1.0 - p_head.detach()) * (
            torch.expm1(log_ratio) - log_ratio
        ).sum(dim=-1) / k_tail

        assert torch.allclose(actual, expected)

    def test_no_tail_matches_topk(self, small_logits):
        """Test k_tail=0 matches the existing top-k objective."""
        teacher, student = small_logits
        k_head = 16

        indices, teacher_log_probs, p_head = select_head_tail_indices(
            teacher, k_head=k_head, k_tail=0
        )
        student_selected = torch.gather(student.detach(), -1, indices)

        actual = subset_k3_kl_from_gathered(
            student_selected,
            teacher_log_probs,
            k_head=k_head,
            k_tail=0,
            p_head=p_head,
        )
        expected = subset_kl_from_gathered(student_selected, teacher_log_probs)

        assert torch.allclose(actual, expected)


class TestComputeSubsetK3KL:
    """Tests for compute_subset_k3_kl convenience function."""

    def test_matches_manual_with_seeded_generator(self, small_logits):
        """Test convenience path matches manual selection with the same seed."""
        teacher, student = small_logits
        k_head, k_tail = 16, 8

        manual_generator = torch.Generator().manual_seed(123)
        indices, teacher_log_probs, p_head = select_head_tail_indices(
            teacher,
            k_head=k_head,
            k_tail=k_tail,
            generator=manual_generator,
        )
        student_selected = torch.gather(student.detach(), -1, indices)
        student_log_normalizer = torch.logsumexp(student.detach().float(), dim=-1)
        expected = subset_k3_kl_from_gathered(
            student_selected,
            teacher_log_probs,
            k_head=k_head,
            k_tail=k_tail,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
        )

        actual_generator = torch.Generator().manual_seed(123)
        actual = compute_subset_k3_kl(
            student.detach(),
            teacher,
            k_head=k_head,
            k_tail=k_tail,
            generator=actual_generator,
        )

        assert torch.allclose(expected, actual)


class TestSubsetImportanceSampledKLFromGathered:
    """Tests for the exact-head plus importance-sampled tail estimator."""

    def test_matches_full_logprob_formula(self, small_logits):
        """Test IS estimator uses full-vocabulary log P and log Q."""
        teacher, student = small_logits
        k_head, k_tail = 16, 8
        generator = torch.Generator().manual_seed(123)

        indices, teacher_log_probs, p_head = select_head_tail_indices(
            teacher, k_head=k_head, k_tail=k_tail, generator=generator
        )
        student_selected = torch.gather(student, -1, indices)
        student_log_normalizer = torch.logsumexp(student.float(), dim=-1)

        actual = subset_is_kl_from_gathered(
            student_selected,
            teacher_log_probs,
            k_head=k_head,
            k_tail=k_tail,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
            reduction="none",
        )

        teacher_full_log_probs = F.log_softmax(teacher.float(), dim=-1)
        student_full_log_probs = F.log_softmax(student.float(), dim=-1)
        head_idx = indices[..., :k_head]
        tail_idx = indices[..., k_head:]
        teacher_head = torch.gather(teacher_full_log_probs, -1, head_idx)
        student_head = torch.gather(student_full_log_probs, -1, head_idx)
        teacher_tail = torch.gather(teacher_full_log_probs, -1, tail_idx)
        student_tail = torch.gather(student_full_log_probs, -1, tail_idx)

        expected = (teacher_head.exp() * (teacher_head - student_head)).sum(dim=-1)
        expected = expected + (1.0 - p_head.detach()) * (
            teacher_tail - student_tail
        ).sum(dim=-1) / k_tail

        assert torch.allclose(actual, expected)

    def test_requires_full_student_normalizer(self, small_logits):
        """Test IS estimator cannot use subset-normalized student probabilities."""
        teacher, student = small_logits
        k_head, k_tail = 16, 8
        indices, teacher_log_probs, p_head = select_head_tail_indices(
            teacher, k_head=k_head, k_tail=k_tail
        )
        student_selected = torch.gather(student, -1, indices)

        with pytest.raises(ValueError, match="student_log_normalizer is required"):
            subset_is_kl_from_gathered(
                student_selected,
                teacher_log_probs,
                k_head=k_head,
                k_tail=k_tail,
                p_head=p_head,
            )

    def test_explicit_mixed_proposal_uses_unbiased_absolute_weights(self, small_logits):
        """Test IS with explicit q uses original P_i / q_i tail weights."""
        teacher, student = small_logits
        k_head, k_tail = 16, 8
        generator = torch.Generator().manual_seed(123)
        indices, teacher_log_probs, p_head, proposal_log_probs = select_head_tail_indices(
            teacher,
            k_head=k_head,
            k_tail=k_tail,
            generator=generator,
            tail_proposal="mixed",
            return_tail_proposal_log_probs=True,
        )
        student_selected = torch.gather(student, -1, indices)
        student_log_normalizer = torch.logsumexp(student.float(), dim=-1)

        actual, diagnostics = subset_is_kl_from_gathered(
            student_selected,
            teacher_log_probs,
            k_head=k_head,
            k_tail=k_tail,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
            tail_proposal_log_probs_selected=proposal_log_probs,
            reduction="none",
            return_diagnostics=True,
        )

        teacher_full_log_probs = F.log_softmax(teacher.float(), dim=-1)
        student_full_log_probs = F.log_softmax(student.float(), dim=-1)
        head_idx = indices[..., :k_head]
        tail_idx = indices[..., k_head:]
        teacher_head = torch.gather(teacher_full_log_probs, -1, head_idx)
        student_head = torch.gather(student_full_log_probs, -1, head_idx)
        teacher_tail = torch.gather(teacher_full_log_probs, -1, tail_idx)
        student_tail = torch.gather(student_full_log_probs, -1, tail_idx)
        expected = (teacher_head.exp() * (teacher_head - student_head)).sum(dim=-1)
        weights = (teacher_tail - proposal_log_probs).exp()
        expected = expected + (weights * (teacher_tail - student_tail)).sum(dim=-1) / k_tail

        assert torch.allclose(actual, expected)
        assert diagnostics["tail_ess_mean"] > 0
        assert diagnostics["tail_max_weight"] > 0




class TestComputeSubsetMCKL:
    """Tests for compute_subset_is_kl convenience function."""

    def test_matches_manual_with_seeded_generator(self, small_logits):
        """Test convenience path matches manual selection with the same seed."""
        teacher, student = small_logits
        k_head, k_tail = 16, 8

        manual_generator = torch.Generator().manual_seed(123)
        indices, teacher_log_probs, p_head = select_head_tail_indices(
            teacher, k_head=k_head, k_tail=k_tail, generator=manual_generator
        )
        student_selected = torch.gather(student.detach(), -1, indices)
        student_log_normalizer = torch.logsumexp(student.detach().float(), dim=-1)
        expected = subset_is_kl_from_gathered(
            student_selected,
            teacher_log_probs,
            k_head=k_head,
            k_tail=k_tail,
            p_head=p_head,
            student_log_normalizer=student_log_normalizer,
        )

        actual_generator = torch.Generator().manual_seed(123)
        actual = compute_subset_is_kl(
            student.detach(),
            teacher,
            k_head=k_head,
            k_tail=k_tail,
            generator=actual_generator,
        )

        assert torch.allclose(expected, actual)




class TestFullKL:
    """Tests for full_kl baseline."""
    
    def test_basic(self, small_logits):
        """Test basic computation."""
        teacher, student = small_logits
        
        loss = full_kl(student, teacher)
        
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestTailProposalExperiments:
    """Tests for repeated-trial tail proposal diagnostics."""

    
    def test_subset_approaches_full(self, logits):
        """Test subset KL approaches full as k increases."""
        teacher, student = logits
        V = teacher.shape[-1]
        
        full_loss = full_kl(student.detach(), teacher).item()
        
        # Higher k should be closer to full
        loss_k64 = compute_subset_kl(student.detach(), teacher, k=64).item()
        loss_k256 = compute_subset_kl(student.detach(), teacher, k=256).item()
        loss_k512 = compute_subset_kl(student.detach(), teacher, k=512).item()
        
        # k=512 should be closer to full than k=64
        assert abs(loss_k512 - full_loss) <= abs(loss_k64 - full_loss)


# =============================================================================
# Class Interface Tests
# =============================================================================

class TestSubsetKLLoss:
    """Tests for SubsetKLLoss class."""
    
    def test_select_indices(self, small_logits):
        """Test select_indices method."""
        teacher, _ = small_logits
        
        loss_fn = SubsetKLLoss(k=32)
        indices, teacher_k = loss_fn.select_indices(teacher)
        
        assert indices.shape[-1] == 32
        assert loss_fn.last_indices is not None
    
    def test_forward_gathered(self, small_logits):
        """Test forward_gathered method."""
        teacher, student = small_logits
        
        loss_fn = SubsetKLLoss(k=32)
        indices, teacher_k = loss_fn.select_indices(teacher)
        student_k = torch.gather(student, -1, indices)
        
        loss = loss_fn.forward_gathered(student_k, teacher_k)
        
        assert torch.isfinite(loss)
    
    def test_forward_convenience(self, small_logits):
        """Test forward() convenience method."""
        teacher, student = small_logits
        
        loss_fn = SubsetKLLoss(k=32)
        loss = loss_fn(student, teacher)
        
        assert torch.isfinite(loss)
    
    def test_two_paths_match(self, small_logits):
        """Test that gathered path matches convenience path."""
        teacher, student = small_logits
        
        loss_fn = SubsetKLLoss(k=32)
        
        # Convenience path
        loss1 = loss_fn(student.detach(), teacher)
        
        # Gathered path
        indices, teacher_k = loss_fn.select_indices(teacher)
        student_k = torch.gather(student.detach(), -1, indices)
        loss2 = loss_fn.forward_gathered(student_k, teacher_k)
        
        assert torch.allclose(loss1, loss2)


class TestKLDivergenceLoss:
    """Tests for KLDivergenceLoss class."""
    
    def test_basic(self, small_logits):
        """Test basic usage."""
        teacher, student = small_logits
        
        loss_fn = KLDivergenceLoss()
        loss = loss_fn(student, teacher)
        
        assert torch.isfinite(loss)
    
    def test_chunked_matches_full(self, small_logits):
        """Test chunked computation matches full."""
        teacher, student = small_logits
        
        loss_full = KLDivergenceLoss()(student.detach(), teacher)
        loss_chunked = KLDivergenceLoss(chunk_size=2)(student.detach(), teacher)
        
        assert torch.allclose(loss_full, loss_chunked, rtol=1e-4)


class TestSubsetK2KLLoss:
    """Tests for SubsetK2KLLoss class."""

    def test_forward_gathered(self, small_logits):
        """Test gathered path."""
        teacher, student = small_logits

        loss_fn = SubsetK2KLLoss(k_head=16, k_tail=8)
        indices, teacher_log_probs, p_head = loss_fn.select_indices(teacher)
        student_selected = torch.gather(student, -1, indices)
        student_log_normalizer = torch.logsumexp(student.float(), dim=-1)
        loss = loss_fn.forward_gathered(
            student_selected,
            teacher_log_probs,
            p_head,
            student_log_normalizer=student_log_normalizer,
        )

        assert torch.isfinite(loss)
        assert loss_fn.last_indices is not None
        assert loss_fn.last_p_head is not None

    def test_forward_convenience(self, small_logits):
        """Test full-logit convenience path."""
        teacher, student = small_logits

        loss_fn = SubsetK2KLLoss(k_head=16, k_tail=8)
        loss = loss_fn(student, teacher)

        assert torch.isfinite(loss)


class TestSubsetK3KLLoss:
    """Tests for SubsetK3KLLoss class."""

    def test_forward_gathered(self, small_logits):
        """Test gathered path."""
        teacher, student = small_logits

        loss_fn = SubsetK3KLLoss(k_head=16, k_tail=8)
        indices, teacher_log_probs, p_head = loss_fn.select_indices(teacher)
        student_selected = torch.gather(student, -1, indices)
        student_log_normalizer = torch.logsumexp(student.float(), dim=-1)
        loss = loss_fn.forward_gathered(
            student_selected,
            teacher_log_probs,
            p_head,
            student_log_normalizer=student_log_normalizer,
        )

        assert torch.isfinite(loss)
        assert loss_fn.last_indices is not None
        assert loss_fn.last_p_head is not None

    def test_forward_convenience(self, small_logits):
        """Test full-logit convenience path."""
        teacher, student = small_logits

        loss_fn = SubsetK3KLLoss(k_head=16, k_tail=8)
        loss = loss_fn(student, teacher)

        assert torch.isfinite(loss)


class TestSubsetImportanceSampledKLLoss:
    """Tests for SubsetImportanceSampledKLLoss class."""

    def test_forward_gathered(self, small_logits):
        """Test gathered path."""
        teacher, student = small_logits

        loss_fn = SubsetImportanceSampledKLLoss(k_head=16, k_tail=8)
        indices, teacher_log_probs, p_head = loss_fn.select_indices(teacher)
        student_selected = torch.gather(student, -1, indices)
        student_log_normalizer = torch.logsumexp(student.float(), dim=-1)
        loss = loss_fn.forward_gathered(
            student_selected,
            teacher_log_probs,
            p_head,
            student_log_normalizer=student_log_normalizer,
        )

        assert torch.isfinite(loss)
        assert loss_fn.last_indices is not None
        assert loss_fn.last_p_head is not None

    def test_forward_convenience(self, small_logits):
        """Test full-logit convenience path."""
        teacher, student = small_logits

        loss_fn = SubsetImportanceSampledKLLoss(k_head=16, k_tail=8)
        loss = loss_fn(student, teacher)

        assert torch.isfinite(loss)








# =============================================================================
# Sampling Tests
# =============================================================================

class TestPPSSampling:
    """Tests for PPS sampling."""
    
    def test_basic(self):
        """Test basic sampling."""
        torch.manual_seed(42)
        N, V = 10, 100
        log_probs = F.log_softmax(torch.randn(N, V), dim=-1)
        
        indices, inc_probs, mask, diag = pps_sample_indices_batched(
            log_probs, k_head=16, k_tail=8
        )
        
        assert indices.shape[0] == N
        assert (indices >= 0).all()
        assert (indices < V).all()
        assert (inc_probs > 0).all()
        assert (inc_probs <= 1).all()
    
    def test_no_tail(self):
        """Test with k_tail=0."""
        N, V = 5, 50
        log_probs = F.log_softmax(torch.randn(N, V), dim=-1)
        
        indices, inc_probs, mask, _ = pps_sample_indices_batched(
            log_probs, k_head=16, k_tail=0
        )
        
        assert indices.shape == (N, 16)
        assert (inc_probs == 1.0).all()  # All deterministic

    def test_pure_importance_sampling(self):
        """Test k_head=0 (pure importance sampling)."""
        N, V = 6, 80
        log_probs = F.log_softmax(torch.randn(N, V), dim=-1)
        
        indices, inc_probs, mask, _ = pps_sample_indices_batched(
            log_probs, k_head=0, k_tail=16, oversample=10
        )
        S = indices.shape[-1]
        
        assert S <= 16
        assert indices.shape == (N, S)
        assert mask.shape == (N, S)
        assert (indices >= 0).all()
        assert (indices < V).all()
        assert (inc_probs > 0).all()
        assert (inc_probs <= 1).all()




# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests."""
    
    def test_efficient_workflow(self, logits):
        """Test the recommended efficient workflow."""
        teacher, student = logits
        k = 128
        
        # Step 1: Select indices
        indices, teacher_k = select_topk_indices(teacher, k)
        
        # Step 2: "Compute" student logits for indices
        # (In real code, this would be lens.forward(hidden, vocab_indices=indices))
        student_k = torch.gather(student, -1, indices)
        
        # Step 3: Compute KL
        loss = subset_kl_from_gathered(student_k, teacher_k)
        
        # Should work and be finite
        assert torch.isfinite(loss)
        
        # Should be able to backprop
        loss.backward()
        assert student.grad is not None
    
    def test_training_loop(self, logits):
        """Simulate a training loop."""
        teacher, student = logits
        
        optimizer = torch.optim.Adam([student], lr=0.01)
        
        initial_loss = compute_subset_kl(student, teacher, k=128).item()
        
        for _ in range(10):
            optimizer.zero_grad()
            loss = compute_subset_kl(student, teacher, k=128)
            loss.backward()
            optimizer.step()
        
        final_loss = compute_subset_kl(student, teacher, k=128).item()
        
        # Student should learn to match teacher
        assert final_loss < initial_loss
    
    def test_memory_scaling_theoretical(self):
        """Verify theoretical memory savings."""
        B, T, V, k = 2, 256, 128000, 256
        
        full_elements = B * T * V
        subset_elements = B * T * k
        
        reduction = full_elements / subset_elements
        assert reduction == 500  # 128000 / 256 = 500x
