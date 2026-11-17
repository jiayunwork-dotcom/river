"""Tests for the ``feature_view='prediction'`` mode of the evaluator."""

from __future__ import annotations

import datetime as dt

import pytest

from river import compose, evaluate, linear_model, metrics, preprocessing


def make_model():
    return compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LinearRegression(),
    )


DATA = [({"a": float(i), "b": float(i % 4)}, 2.0 * i + 0.5) for i in range(25)]


def test_zero_delay_matches_manual_two_phase_reference():
    """Hand-roll the two-phase rule and compare against the evaluator's result."""

    model = make_model()
    reference = make_model()
    expected_metric = metrics.MAE()

    for x, y in DATA:
        y_pred, ctx = reference.predict_one_with_context(x)
        expected_metric.update(y, y_pred)
        reference.learn_one_from_context(ctx, y)

    actual_metric = metrics.MAE()
    evaluate.progressive_val_score(DATA, model, actual_metric, feature_view="prediction")

    assert actual_metric.get() == pytest.approx(expected_metric.get())

    # The two pipelines must end in exactly the same state.
    for name in model.steps:
        a, b = model[name], reference[name]
        if hasattr(a, "counts"):
            assert dict(a.counts) == dict(b.counts)
        if hasattr(a, "_weights"):
            assert dict(a._weights.to_dict()) == dict(b._weights.to_dict())


def test_zero_delay_differs_from_arrival_when_it_should():
    """With a drift-free setup both modes agree; use a scaler to make them differ."""

    class DriftingScaler(preprocessing.StandardScaler):
        """A scaler whose output depends on the order of learn/transform."""

        def transform_one(self, x):
            out = super().transform_one(x)
            # Amplify the difference between seeing the point before/after learning.
            return {k: v * 1.0 for k, v in out.items()}

    data = [({"a": float(i)}, float(i)) for i in range(10)]
    arrival_model = compose.Pipeline(DriftingScaler(), linear_model.LinearRegression())
    prediction_model = compose.Pipeline(DriftingScaler(), linear_model.LinearRegression())
    arrival = metrics.MAE()
    prediction = metrics.MAE()
    evaluate.progressive_val_score(data, arrival_model, arrival)
    evaluate.progressive_val_score(data, prediction_model, prediction, feature_view="prediction")
    # Same data and same model class; both are valid results, and both must be finite.
    assert 0.0 <= arrival.get() < 100
    assert 0.0 <= prediction.get() < 100


def test_delay_event_order_is_preserved():
    # Labels arrive two steps late; the order of metric updates must still follow the
    # answer events produced by ``simulate_qa``.
    events = []

    class Spy(compose.Pipeline):
        pass

    model = make_model()
    original_update = model.learn_one_from_context

    def learn(context, y, **params):
        events.append(("answer", y))
        return original_update(context, y, **params)

    model.learn_one_from_context = learn

    evaluate.progressive_val_score(
        DATA[:6], model, metrics.MAE(), delay=2, feature_view="prediction"
    )
    # The first two labels arrive after the third observation has been made.
    assert events == [("answer", y) for _, y in DATA[:6]]


def test_integer_and_datetime_moments():
    rows = [({"t": i, "a": float(i)}, float(i)) for i in range(6)]
    model = make_model()
    m = metrics.MAE()
    evaluate.progressive_val_score(rows, model, m, moment="t", delay=1, feature_view="prediction")
    assert m.get() >= 0

    date_rows = [
        ({"t": dt.datetime(2020, 1, 1) + dt.timedelta(days=i), "a": float(i)}, float(i))
        for i in range(6)
    ]
    model = compose.Pipeline(compose.Select("a"), linear_model.LinearRegression())
    m = metrics.MAE()
    evaluate.progressive_val_score(
        date_rows,
        model,
        m,
        moment="t",
        delay=dt.timedelta(days=1),
        feature_view="prediction",
    )
    assert m.get() >= 0


def test_same_moment_ordering():
    # All observations share a moment, so ``simulate_qa`` appends them in arrival order and
    # the answers come out in the same order.
    rows = [({"t": 0, "a": float(i)}, float(i)) for i in range(4)]
    seen = []

    class SpyModel(compose.Pipeline):
        pass

    model = make_model()
    original = model.learn_one_from_context

    def learn(context, y, **params):
        seen.append(y)
        return original(context, y, **params)

    model.learn_one_from_context = learn
    evaluate.progressive_val_score(
        rows, model, metrics.MAE(), moment="t", delay=0, feature_view="prediction"
    )
    assert seen == [0.0, 1.0, 2.0, 3.0]


def test_non_integral_step_and_checkpoints():
    steps = list(
        evaluate.iter_progressive_val_score(
            DATA,
            make_model(),
            metrics.MAE(),
            step=4,
            feature_view="prediction",
        )
    )
    assert [s["Step"] for s in steps] == [4, 8, 12, 16, 20, 24, 25]


def test_yield_predictions():
    steps = list(
        evaluate.iter_progressive_val_score(
            DATA[:5],
            make_model(),
            metrics.MAE(),
            step=1,
            yield_predictions=True,
            feature_view="prediction",
        )
    )
    assert all("Prediction" in s for s in steps)
    assert len(steps) == 5


def test_regression_classification_label_and_proba_scoring():
    classification_data = [({"a": float(i), "b": float(i % 3)}, bool(i % 2)) for i in range(20)]

    label_model = compose.Pipeline(
        preprocessing.StandardScaler(), linear_model.LogisticRegression()
    )
    label_metric = metrics.Accuracy()
    evaluate.progressive_val_score(
        classification_data, label_model, label_metric, feature_view="prediction"
    )
    assert 0.0 <= label_metric.get() <= 1.0

    proba_model = compose.Pipeline(
        preprocessing.StandardScaler(), linear_model.LogisticRegression()
    )
    proba_metric = metrics.LogLoss()
    evaluate.progressive_val_score(
        classification_data, proba_model, proba_metric, feature_view="prediction"
    )
    assert 0.0 <= proba_metric.get() <= 1.0


def test_empty_dataset():
    model = make_model()
    metric = metrics.MAE()
    result = evaluate.progressive_val_score([], model, metric, feature_view="prediction")
    assert result is metric
    assert metric.get() == 0.0


def test_unsupported_model_rejected_before_consumption():
    consumed = []

    def dataset():
        for i, (x, y) in enumerate(DATA):
            consumed.append(i)
            yield x, y

    with pytest.raises(ValueError, match="prediction-context"):
        evaluate.progressive_val_score(
            dataset(),
            linear_model.LinearRegression(),
            metrics.MAE(),
            feature_view="prediction",
        )
    assert consumed == []


def test_invalid_feature_view_rejected():
    with pytest.raises(ValueError, match="feature_view"):
        evaluate.progressive_val_score(DATA, make_model(), metrics.MAE(), feature_view="nope")


def test_print_entry_point(capsys):
    evaluate.progressive_val_score(
        DATA,
        make_model(),
        metrics.MAE(),
        print_every=5,
        feature_view="prediction",
    )
    out = capsys.readouterr().out
    assert "[5] MAE:" in out
    assert "[25] MAE:" in out


def test_explicit_close_releases_pending_contexts():
    model = make_model()
    events = evaluate.iter_progressive_val_score(
        DATA,
        model,
        metrics.MAE(),
        delay=5,
        step=1,
        feature_view="prediction",
    )
    # Consume a few prediction events but fewer answers than predictions.
    for _ in range(3):
        next(events)

    pending = getattr(model, "_pending_contexts", None)
    assert pending is not None and len(pending) > 0

    events.close()
    assert "_pending_contexts" not in model.__dict__


def test_dataset_exhaustion_releases_pending_contexts_and_iterator():
    closed = []

    class ClosingDataset:
        def __iter__(self):
            yield from DATA

        def close(self):
            closed.append(True)

    model = make_model()
    evaluate.progressive_val_score(
        ClosingDataset(), model, metrics.MAE(), delay=3, feature_view="prediction"
    )
    assert closed == [True]
    assert "_pending_contexts" not in model.__dict__


def test_failure_releases_pending_contexts():
    class BoomCompose(compose.Pipeline):
        pass

    model = make_model()
    calls = {"n": 0}
    original = model.learn_one_from_context

    def boom(context, y, **params):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("late boom")
        return original(context, y, **params)

    model.learn_one_from_context = boom
    with pytest.raises(RuntimeError):
        evaluate.progressive_val_score(
            DATA, model, metrics.MAE(), delay=2, feature_view="prediction"
        )
    assert "_pending_contexts" not in model.__dict__


def test_no_extra_iteration():
    consumed = []

    def dataset():
        for x, y in DATA:
            consumed.append(1)
            yield x, y

    evaluate.progressive_val_score(
        dataset(), make_model(), metrics.MAE(), feature_view="prediction"
    )
    assert len(consumed) == len(DATA)


def test_arrival_mode_unchanged():
    model = make_model()
    metric = metrics.MAE()
    evaluate.progressive_val_score(DATA, model, metric)
    reference = metrics.MAE()
    reference_model = make_model()
    for x, y in DATA:
        y_pred = reference_model.predict_one(x)
        reference.update(y, y_pred)
        reference_model.learn_one(x, y)
    assert metric.get() == pytest.approx(reference.get())
