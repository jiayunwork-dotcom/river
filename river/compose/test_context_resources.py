"""Resource and scale tests for the prediction-context protocol.

These are structural tests, not micro-benchmarks. They stream at least 100k observations
and check that the material retained by the pipeline is bounded by the number of *pending*
observations, not by the number of observations seen. Measurements (pending count, retained
payload, peak traced memory and wall time) are printed so they can be quoted in the docs.
"""

from __future__ import annotations

import gc
import time
import tracemalloc
import weakref

import pytest

from river import base, compose, evaluate, linear_model, metrics, preprocessing


class FixedRecorder:
    """A transformer with bounded state, used in place of a growing model."""

    _supervised = False

    def __init__(self):
        self.n = 0

    def learn_one(self, x):
        self.n += 1

    def transform_one(self, x):
        return {"x": x["x"], "n": self.n}


class FixedSupervisedRecorder(base.SupervisedTransformer):
    """A supervised transformer that snapshots its input, with bounded state.

    Including one of these makes the context retain the *input* it saw, which is what the
    feature-width measurements probe.
    """

    def __init__(self):
        self.n = 0

    def learn_one(self, x, y):
        self.n += 1

    def transform_one(self, x):
        return {**x, "seen": self.n}


class FixedTerminal(base.Regressor):
    def __init__(self):
        self.n = 0

    def learn_one(self, x, y):
        self.n += 1

    def predict_one(self, x):
        return 0.0


def _wide_pipeline(width, depth):
    """A union tree with ``depth`` layers, each with ``width`` branches.

    Every branch ends with a supervised transformer, so each pending context stores one
    input per supervised leaf, which is what the depth measurement probes.
    """
    block = compose.TransformerUnion(
        *[
            (
                f"leaf_{j}",
                compose.Pipeline(
                    compose.Select("x"),
                    preprocessing.StandardScaler(),
                    FixedSupervisedRecorder(),
                ),
            )
            for j in range(width)
        ]
    )
    for d in range(depth - 1):
        block = compose.TransformerUnion(*[(f"layer_{d}_{j}", block.clone()) for j in range(width)])
    return compose.Pipeline(block, linear_model.LinearRegression())


def _run(n_observations, max_pending, width, model_factory):
    model = model_factory()
    held = []
    peak_payload = 0
    tracemalloc.start()
    started = time.perf_counter()
    try:
        for i in range(n_observations):
            x = {"x": float(i)}
            for w in range(width - 1):
                x[f"pad_{w}"] = float(i + w)
            _, ctx = model.predict_one_with_context(x)
            held.append(ctx)
            if len(held) > max_pending:
                model.discard_context(held.pop(0))
            if i % 200 == 0:
                pending = model._pending_contexts
                peak_payload = max(peak_payload, sum(c._payload_size() for c in pending))
        pending_count = len(model._pending_contexts)
        _, peak_memory = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    elapsed = time.perf_counter() - started

    for ctx in held:
        model.discard_context(ctx)
    assert "_pending_contexts" not in model.__dict__

    return {
        "pending": pending_count,
        "payload": peak_payload,
        "peak_memory": peak_memory,
        "elapsed": elapsed,
    }


@pytest.mark.parametrize("max_pending", [8, 128])
@pytest.mark.parametrize("width", [2, 16])
def test_stream_100k_bounded(max_pending, width):
    result = _run(
        n_observations=100_000,
        max_pending=max_pending,
        width=width,
        model_factory=lambda: compose.Pipeline(FixedSupervisedRecorder(), FixedTerminal()),
    )
    print(
        f"\npending={max_pending} width={width} "
        f"retained_pending={result['pending']} "
        f"peak_payload={result['payload']}B "
        f"peak_memory={result['peak_memory'] / 1024:.1f}KiB "
        f"elapsed={result['elapsed']:.2f}s"
    )
    assert result["pending"] == max_pending
    # Retention scales with the number of pending observations and the width of the
    # *input*, not with the 100k observations processed.
    assert result["payload"] < max_pending * width * 2048
    assert result["peak_memory"] < 64 * 1024 * 1024
    assert result["elapsed"] < 120


@pytest.mark.parametrize("depth", [1, 3])
def test_branch_depth_cost_is_structural(depth):
    result = _run(
        n_observations=20_000,
        max_pending=16,
        width=2,
        model_factory=lambda: _wide_pipeline(width=2, depth=depth),
    )
    print(
        f"\ndepth={depth} retained_pending={result['pending']} "
        f"peak_payload={result['payload']}B "
        f"peak_memory={result['peak_memory'] / 1024:.1f}KiB "
        f"elapsed={result['elapsed']:.2f}s"
    )
    assert result["pending"] == 16
    # A deeper tree costs more per prediction (more leaves run) but retention stays tied to
    # the pending count: each pending context stores one small input per supervised leaf,
    # not a copy of the tree or of the observations.
    leaves = 2**depth
    assert result["payload"] < 16 * leaves * 1024


def test_pending_material_is_bounded_by_pending_count():
    model = compose.Pipeline(
        compose.Select("a", "b", "c"),
        preprocessing.StandardScaler(),
        linear_model.LinearRegression(),
    )
    held = []
    for i in range(200):
        _, ctx = model.predict_one_with_context(
            {"a": float(i), "b": float(i % 7), "c": float(i % 3), "noise": "x" * 50}
        )
        held.append(ctx)
        if len(held) > 8:
            model.discard_context(held.pop(0))

    pending = list(model._pending_contexts)
    assert len(pending) == 8
    assert sum(ctx._payload_size() for ctx in pending) < 8 * 4096

    for ctx in held:
        model.discard_context(ctx)
    assert "_pending_contexts" not in model.__dict__


def test_finished_contexts_release_payload():
    model = compose.Pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression())
    _, ctx = model.predict_one_with_context({"a": 1.0, "b": 2.0})
    assert ctx._payload_size() > 0
    model.learn_one_from_context(ctx, 1.0)
    assert ctx._payload_size() == 0
    assert not ctx._is_pending


def test_failed_contexts_release_payload():
    class Boom(base.Regressor):
        def predict_one(self, x):
            return 0.0

        def learn_one(self, x, y):
            raise RuntimeError("nope")

    pipeline = compose.Pipeline(compose.Select("a"), Boom())
    _, ctx = pipeline.predict_one_with_context({"a": 1.0})
    with pytest.raises(RuntimeError):
        pipeline.learn_one_from_context(ctx, 1.0)
    assert ctx._payload_size() == 0


def test_repeated_create_learn_discard_and_hold_does_not_accumulate():
    model = compose.Pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression())
    finished = []
    for i in range(500):
        # Learned and discarded contexts stay in the caller's hands.
        _, learned = model.predict_one_with_context({"a": float(i), "b": float(i)})
        model.learn_one_from_context(learned, float(i))
        finished.append(learned)

        _, discarded = model.predict_one_with_context({"a": float(i), "b": float(i)})
        model.discard_context(discarded)
        finished.append(discarded)

        # No table survives once nothing is pending.
        assert "_pending_contexts" not in model.__dict__

    assert all(not ctx._is_pending for ctx in finished)
    assert all(ctx._payload_size() == 0 for ctx in finished)
    assert "_pending_contexts" not in model.__dict__


def test_no_history_table_after_feedback():
    model = compose.Pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression())
    for i in range(100):
        _, ctx = model.predict_one_with_context({"a": float(i)})
        model.learn_one_from_context(ctx, float(i))
    assert "_pending_contexts" not in model.__dict__


def test_dropped_context_is_garbage_collected():
    model = compose.Pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression())
    _, ctx = model.predict_one_with_context({"a": 1.0})
    ref = weakref.ref(ctx)
    del ctx
    gc.collect()
    assert ref() is None
    assert "_pending_contexts" not in model.__dict__


def test_evaluator_close_frees_pending_contexts():
    data = [({"a": float(i)}, float(i)) for i in range(50)]
    model = compose.Pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression())
    events = evaluate.iter_progressive_val_score(
        data, model, metrics.MAE(), delay=10, feature_view="prediction"
    )
    for _ in range(4):
        next(events)
    assert model.__dict__.get("_pending_contexts")
    events.close()
    assert "_pending_contexts" not in model.__dict__


def test_failure_frees_pending_contexts():
    data = [({"a": float(i)}, float(i)) for i in range(50)]
    model = compose.Pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression())
    original = model.learn_one_from_context
    calls = {"n": 0}

    def boom(context, y, **params):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("late boom")
        return original(context, y, **params)

    model.learn_one_from_context = boom
    with pytest.raises(RuntimeError):
        evaluate.progressive_val_score(
            data, model, metrics.MAE(), delay=5, feature_view="prediction"
        )
    assert "_pending_contexts" not in model.__dict__
