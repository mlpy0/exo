# type: ignore
"""Per-entry logits processors in the patched batch step.

``GenerationBatch`` normalizes the processor list to ``[None] * len(uids)`` when a
merged-in batch brings none of its own, so a batch that mixes a request with
penalties and one without holds ``None`` next to a real processor list.
"""

from types import SimpleNamespace

import mlx.core as mx

from exo.worker.engines.mlx.patches.opt_batch_gen import _patched_step

VOCAB = 8


def _boost(token_id: int):
    """Logits processor that makes ``token_id`` the argmax for its row."""

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        bias = mx.zeros(logits.shape)
        bias[:, token_id] = 1.0
        return logits + bias

    return processor


def _batch(logits_processors) -> SimpleNamespace:
    """Two-request batch over flat logits, so the argmax is entirely processor-driven."""
    logits = mx.zeros((2, 1, VOCAB))
    return SimpleNamespace(
        model=lambda inputs, cache=None: logits,
        prompt_cache=None,
        uids=[0, 1],
        tokens=[[1, 2], [3, 4]],
        logits_processors=logits_processors,
        samplers=None,
        fallback_sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        _next_tokens=mx.array([1, 3]),
        _next_logprobs=[],
    )


def test_step_skips_none_entries_and_applies_the_rest() -> None:
    batch = _batch([None, [_boost(5)]])

    _patched_step(batch)

    assert batch._next_tokens.tolist() == [0, 5]


def test_step_handles_an_all_none_processor_list() -> None:
    batch = _batch([None, None])

    _patched_step(batch)

    assert batch._next_tokens.tolist() == [0, 0]
