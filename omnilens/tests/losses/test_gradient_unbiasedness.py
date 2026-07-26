# tests/losses/test_gradient_unbiasedness.py
"""The Top-k+IS estimator (mode="is") must give unbiased gradients of the
full KL w.r.t. student parameters (paper Theorem 1). Top-k renormalization,
which truncates the partition function, must not.

This is the numerical companion to the unbiasedness proof: only the KL
summands are subsampled; the student log-partition is exact over the full
vocabulary, so every logit receives gradient through the partition term.
"""

import torch

from omnilens.losses.subset_kl import SubsetKLLoss


def _exact_full_kl_grad(h, Ws, teacher_logits):
    sl = h @ Ws
    p = torch.softmax(teacher_logits, -1)
    q_log = torch.log_softmax(sl, -1)
    # xlogy(0, 0) = 0, so tokens whose teacher prob underflows fp32
    # contribute nothing instead of NaN (0 * log 0).
    kl = (torch.special.xlogy(p, p) - p * q_log).sum(-1).mean()
    kl.backward()
    g = Ws.grad.clone()
    Ws.grad = None
    return g


def _mean_estimator_grad(loss_fn, h, Ws, teacher_logits, n):
    g = torch.zeros_like(Ws)
    for _ in range(n):
        loss = loss_fn(h @ Ws, teacher_logits)
        loss.backward()
        g += Ws.grad
        Ws.grad = None
    return g / n


def _setup(seed=0, B=1, T=8, V=200, d=16):
    torch.manual_seed(seed)
    teacher_logits = torch.randn(B, T, V) * 2.0
    h = torch.randn(B, T, d)
    Ws = torch.nn.Parameter(torch.randn(d, V) * 0.3)
    return h, Ws, teacher_logits


def test_topk_is_gradients_unbiased():
    h, Ws, teacher_logits = _setup()
    g_full = _exact_full_kl_grad(h, Ws, teacher_logits)

    is_loss = SubsetKLLoss(k=20, mode="is", k_tail=10)
    # n=4000 matches the numerical check quoted in the paper's theory appendix.
    g_is = _mean_estimator_grad(is_loss, h, Ws, teacher_logits, n=4000)

    rel_bias = ((g_is - g_full).norm() / g_full.norm()).item()
    cosine = torch.nn.functional.cosine_similarity(
        g_is.flatten(), g_full.flatten(), dim=0
    ).item()
    assert rel_bias < 0.02, f"Top-k+IS gradient bias too large: {rel_bias:.4f}"
    assert cosine > 0.999, f"Top-k+IS gradient direction off: cos={cosine:.5f}"


def test_topk_is_gradients_unbiased_llama_vocab_extreme_logits():
    """Same claim at LLaMA-scale vocabulary (128,256) with extreme teacher
    logits, so much of the tail underflows fp32 softmax. Exercises the
    numerical-guard path: guards must not modify the teacher distribution."""
    h, Ws, teacher_logits = _setup(V=128256, T=4)
    teacher_logits = teacher_logits * 8.0  # spread ~ +/- 25 nats
    g_full = _exact_full_kl_grad(h, Ws, teacher_logits)

    is_loss = SubsetKLLoss(k=512, mode="is", k_tail=1024)
    g_is = _mean_estimator_grad(is_loss, h, Ws, teacher_logits, n=500)

    rel_bias = ((g_is - g_full).norm() / g_full.norm()).item()
    cosine = torch.nn.functional.cosine_similarity(
        g_is.flatten(), g_full.flatten(), dim=0
    ).item()
    assert rel_bias < 0.05, f"Top-k+IS gradient bias too large: {rel_bias:.4f}"
    assert cosine > 0.999, f"Top-k+IS gradient direction off: cos={cosine:.5f}"


def test_topk_renormalized_gradients_biased():
    """Sanity check that the test is sharp: plain top-k is visibly biased."""
    h, Ws, teacher_logits = _setup()
    g_full = _exact_full_kl_grad(h, Ws, teacher_logits)

    tk_loss = SubsetKLLoss(k=30, mode="topk")
    g_tk = _mean_estimator_grad(tk_loss, h, Ws, teacher_logits, n=20)

    rel_bias = ((g_tk - g_full).norm() / g_full.norm()).item()
    assert rel_bias > 0.2, (
        "top-k renormalized KL unexpectedly matched full-KL gradients; "
        f"rel_bias={rel_bias:.4f}"
    )
