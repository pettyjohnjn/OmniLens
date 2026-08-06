"""Exact full-vocabulary log-partition without an fp32 [N, V] copy.

``full_logits.float().logsumexp(-1)`` materializes a second full-vocab
tensor, and autograd keeps that fp32 copy alive until backward. Here the
reduction streams over vocabulary chunks with fp32 accumulation (the same
reduction, combined via logaddexp), and backward recomputes the softmax
chunk-by-chunk, so only the original low-precision logits and the [N, 1]
result survive the forward pass.

Paper mapping: this is the exact log-partition computation described in
Sec. 4.2 ('streaming the log-sum-exp reduction over vocabulary chunks in
fp32'), which keeps the Top-k+IS estimator's student normalization exact
(Theorem 1 applies to the implemented objective).
"""
from __future__ import annotations

import torch

_CHUNK = 8192


class _StreamingLogsumexp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor) -> torch.Tensor:
        lse = None
        for s in range(0, logits.shape[1], _CHUNK):
            part = logits[:, s:s + _CHUNK].float().logsumexp(dim=1, keepdim=True)
            lse = part if lse is None else torch.logaddexp(lse, part)
        ctx.save_for_backward(logits, lse)
        return lse

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
        logits, lse = ctx.saved_tensors
        grad = torch.empty_like(logits)
        for s in range(0, logits.shape[1], _CHUNK):
            p = torch.exp(logits[:, s:s + _CHUNK].float() - lse)
            grad[:, s:s + _CHUNK] = (p * grad_out).to(grad.dtype)
        return grad


def streaming_logsumexp(full_logits: torch.Tensor) -> torch.Tensor:
    """fp32 logsumexp over dim 1 of a [N, V] tensor, streamed in vocab chunks."""
    return _StreamingLogsumexp.apply(full_logits)
