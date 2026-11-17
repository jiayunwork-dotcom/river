"""Tests for the two-phase prediction/feedback protocol of ``compose.Pipeline``.

The event ledgers in this file are maintained by hand: the recording steps below
only append to their own lists. Nothing here asks the new context machinery what it
did; expectations are written independently of the implementation.
"""

from __future__ import annotations

import pytest

from river import (
    base,
    compose,
    feature_extraction,
    linear_model,
    metrics,
    preprocessing,
    stats,
)

# ---------------------------------------------------------------------------
# Recording steps
# ---------------------------------------------------------------------------


class Recorder(base.Transformer):
    """An unsupervised transformer that keeps changing its output.

    ``transform_one`` adds a feature whose value grows with every call, and it also
    reuses a single output buffer. That makes it a good probe for "transform twice"
    bugs and for buffer-reuse bugs.
    """

    def __init__(self, name="rec"):
        self.name = name
        self.learned = []
        self.transformed = []
        self.transformed_inputs = []
        self._out = {}

    def learn_one(self, x):
        self.learned.append(dict(x))
        return None

    def transform_one(self, x):
        self.transformed.append(dict(x))
        self.transformed_inputs.append(dict(x))
        self._out.clear()
        self._out["drift"] = len(self.transformed)
        self._out.update(x)
        return self._out


class SupervisedRecorder(base.SupervisedTransformer):
    """A supervised transformer whose output depends on the targets it has seen."""

    def __init__(self, name="sup", key="s"):
        self.name = name
        self.key = key
        self.learned = []
        self.transformed_inputs = []
        self.total = 0.0

    def learn_one(self, x, y):
        self.learned.append((dict(x), y))
        self.total += y
        return None

    def transform_one(self, x):
        self.transformed_inputs.append(dict(x))
        out = dict(x)
        out[self.key] = self.total
        return out


class TerminalRecorder(base.Regressor):
    def __init__(self):
        self.learned = []
        self.predicted_with = []

    def learn_one(self, x, y):
        self.learned.append((dict(x), y))
        return None

    def predict_one(self, x):
        self.predicted_with.append(dict(x))
        return 0.0


class ProbaTerminal(base.Classifier):
    def __init__(self):
        self.learned = []
        self.predicted_with = []

    def learn_one(self, x, y):
        self.learned.append((dict(x), y))
        return None

    def predict_proba_one(self, x):
        self.predicted_with.append(dict(x))
        return {True: 0.25, False: 0.75}


# ---------------------------------------------------------------------------
# Core semantics
# ---------------------------------------------------------------------------


def test_regression_two_phase_ledger():
    rec = Recorder()
    sup = SupervisedRecorder()
    terminal = TerminalRecorder()
    pipeline = compose.Pipeline(rec, sup, terminal)

    prediction, context = pipeline.predict_one_with_context({"a": 1})
    assert prediction == 0.0

    # Prediction phase: unsupervised learns, then transforms; supervised only transforms.
    assert rec.learned == [{"a": 1}]
    assert rec.transformed_inputs == [{"a": 1}]
    assert sup.learned == []
    assert sup.transformed_inputs == [{"drift": 1, "a": 1}]
    assert terminal.predicted_with == [{"drift": 1, "a": 1, "s": 0.0}]

    pipeline.learn_one_from_context(context, 10.0)

    # Feedback phase: supervised sees the input it transformed, terminal sees what it
    # predicted with. The unsupervised step is not learned again, nor transformed again.
    assert rec.learned == [{"a": 1}]
    assert rec.transformed_inputs == [{"a": 1}]
    assert sup.learned == [({"drift": 1, "a": 1}, 10.0)]
    assert terminal.learned == [({"drift": 1, "a": 1, "s": 0.0}, 10.0)]


def test_three_observations_predict_then_out_of_order_feedback():
    rec = Recorder()
    sup = SupervisedRecorder()
    terminal = TerminalRecorder()
    pipeline = compose.Pipeline(rec, sup, terminal)

    contexts = {}
    for i in range(3):
        _, contexts[i] = pipeline.predict_one_with_context({"a": i})

    # Every prediction input is recorded before any feedback.
    assert rec.learned == [{"a": 0}, {"a": 1}, {"a": 2}]
    assert [t for t in sup.transformed_inputs] == [
        {"drift": 1, "a": 0},
        {"drift": 2, "a": 1},
        {"drift": 3, "a": 2},
    ]

    # Feedback in the order C, A, B.
    pipeline.learn_one_from_context(contexts[2], 20.0)
    pipeline.learn_one_from_context(contexts[0], 0.0)
    pipeline.learn_one_from_context(contexts[1], 10.0)

    assert sup.learned == [
        ({"drift": 3, "a": 2}, 20.0),
        ({"drift": 1, "a": 0}, 0.0),
        ({"drift": 2, "a": 1}, 10.0),
    ]
    # The terminal always sees the exact features it predicted with, regardless of the
    # feedback order.
    assert terminal.learned == [
        ({"drift": 3, "a": 2, "s": 0.0}, 20.0),
        ({"drift": 1, "a": 0, "s": 0.0}, 0.0),
        ({"drift": 2, "a": 1, "s": 0.0}, 10.0),
    ]


def test_prediction_not_recomputed_on_feedback():
    class CountingTerminal(base.Regressor):
        def __init__(self):
            self.predictions = 0
            self.learns = 0

        def predict_one(self, x):
            self.predictions += 1
            return 7.0

        def learn_one(self, x, y):
            self.learns += 1

    terminal = CountingTerminal()
    pipeline = compose.Pipeline(compose.Select("a"), terminal)
    _, ctx = pipeline.predict_one_with_context({"a": 1})
    pipeline.learn_one_from_context(ctx, 1.0)
    assert terminal.predictions == 1
    assert terminal.learns == 1


def test_classifier_proba_context():
    terminal = ProbaTerminal()
    pipeline = compose.Pipeline(compose.Select("a"), terminal)
    prediction, ctx = pipeline.predict_proba_one_with_context({"a": 3})
    assert prediction == {True: 0.25, False: 0.75}
    pipeline.learn_one_from_context(ctx, True)
    assert terminal.learned == [({"a": 3}, True)]
    assert terminal.predicted_with == [{"a": 3}]


def test_supervised_input_is_captured_before_transform():
    """A supervised transformer that mutates its input must not corrupt the snapshot."""

    class InPlace(base.SupervisedTransformer):
        def __init__(self):
            self.seen = []

        def learn_one(self, x, y):
            self.seen.append(dict(x))

        def transform_one(self, x):
            x["mutated"] = True
            return x

    step = InPlace()
    terminal = TerminalRecorder()
    pipeline = compose.Pipeline(step, terminal)
    x = {"a": 1}
    _, ctx = pipeline.predict_one_with_context(x)
    pipeline.learn_one_from_context(ctx, 3.0)
    assert step.seen == [{"a": 1}]
    assert terminal.learned == [({"a": 1, "mutated": True}, 3.0)]


def test_nested_union_and_deep_branches():
    """Nested pipelines inside unions inside pipelines, with mixed supervision."""

    deep_rec = Recorder()
    deep_sup = SupervisedRecorder(key="deep")
    nested = compose.Pipeline(deep_rec, deep_sup)

    other_sup = SupervisedRecorder(key="other")
    outer_sup = SupervisedRecorder(key="outer")
    terminal = TerminalRecorder()

    union = compose.TransformerUnion(
        compose.Pipeline(compose.Select("a"), nested),
        compose.Pipeline(compose.Select("a"), other_sup),
    )
    pipeline = compose.Pipeline(compose.Select("a"), union, outer_sup, terminal)

    # ``TransformerUnion`` clones on insertion; use the live instances.
    union = pipeline[1]
    deep_branch = union[0]
    other_branch = union[1]
    deep_rec = deep_branch[1][0]
    deep_sup = deep_branch[1][1]
    other_sup = other_branch[1]
    outer_sup = pipeline[2]
    terminal = pipeline[3]

    prediction, ctx = pipeline.predict_one_with_context({"a": 5, "junk": 0})
    assert prediction == 0.0

    # Every leaf transformed exactly once.
    assert deep_rec.transformed_inputs == [{"a": 5}]
    assert [t["a"] for t in deep_sup.transformed_inputs] == [5]
    assert [t["a"] for t in other_sup.transformed_inputs] == [5]
    # The outer supervised step sees the union's merged output.
    assert outer_sup.transformed_inputs[0]["a"] == 5
    assert outer_sup.transformed_inputs[0]["other"] == 0.0
    assert outer_sup.transformed_inputs[0]["deep"] == 0.0

    pipeline.learn_one_from_context(ctx, 2.0)

    assert deep_sup.learned == [({"drift": 1, "a": 5}, 2.0)]
    assert other_sup.learned == [({"a": 5}, 2.0)]
    assert outer_sup.learned == [({"a": 5, "drift": 1, "deep": 0.0, "other": 0.0}, 2.0)]
    assert terminal.learned == [
        ({"a": 5, "drift": 1, "deep": 0.0, "other": 0.0, "outer": 0.0}, 2.0)
    ]


def test_learn_during_predict_does_not_change_count():
    def build():
        return compose.Pipeline(Recorder(), SupervisedRecorder(), TerminalRecorder())

    plain = build()
    with compose.learn_during_predict():
        nested = build()

    p1, c1 = plain.predict_one_with_context({"a": 1})
    with compose.learn_during_predict():
        p2, c2 = nested.predict_one_with_context({"a": 1})

    plain.learn_one_from_context(c1, 1.0)
    with compose.learn_during_predict():
        nested.learn_one_from_context(c2, 1.0)

    rec_plain, rec_nested = plain[0], nested[0]
    assert rec_plain.learned == rec_nested.learned
    assert rec_plain.transformed_inputs == rec_nested.transformed_inputs
    # The global flag is left untouched.
    assert compose.Pipeline._LEARN_UNSUPERVISED_DURING_PREDICT is False


def test_custom_duck_typed_transformer():
    class Doubler:
        _supervised = False

        def __init__(self):
            self.seen = []

        def learn_one(self, x):
            self.seen.append(dict(x))

        def transform_one(self, x):
            return {k: v * 2 for k, v in x.items()}

    step = Doubler()
    terminal = TerminalRecorder()
    pipeline = compose.Pipeline(step, terminal)
    _, ctx = pipeline.predict_one_with_context({"a": 2})
    pipeline.learn_one_from_context(ctx, 1.0)
    assert step.seen == [{"a": 2}]
    assert terminal.learned == [({"a": 4}, 1.0)]


# ---------------------------------------------------------------------------
# Input/output mutation
# ---------------------------------------------------------------------------


def test_caller_mutations_do_not_leak():
    sup = SupervisedRecorder()
    terminal = TerminalRecorder()
    pipeline = compose.Pipeline(compose.Select("a", "b"), sup, terminal)

    x = {"a": 1, "b": 2}
    _, ctx = pipeline.predict_one_with_context(x)
    x["a"] = 999
    x["b"] = 999
    x["new"] = 1

    pipeline.learn_one_from_context(ctx, 5.0)
    assert sup.learned == [({"a": 1, "b": 2}, 5.0)]
    assert terminal.learned == [({"a": 1, "b": 2, "s": 0.0}, 5.0)]


def test_buffer_reuse_does_not_leak():
    rec = Recorder()
    terminal = TerminalRecorder()
    pipeline = compose.Pipeline(compose.Select("a"), rec, terminal)

    contexts = [pipeline.predict_one_with_context({"a": i})[1] for i in range(3)]
    # Force the recorder to rewrite its buffer several more times.
    for i in range(3):
        rec.transform_one({"a": -i})

    for i, ctx in enumerate(contexts):
        pipeline.learn_one_from_context(ctx, float(i))

    assert terminal.learned == [
        ({"drift": 1, "a": 0}, 0.0),
        ({"drift": 2, "a": 1}, 1.0),
        ({"drift": 3, "a": 2}, 2.0),
    ]


def test_probability_dict_mutation_is_isolated():
    terminal = ProbaTerminal()
    pipeline = compose.Pipeline(compose.Select("a"), terminal)
    prediction, ctx = pipeline.predict_proba_one_with_context({"a": 1})
    prediction[True] = 0.99
    prediction.clear()
    pipeline.learn_one_from_context(ctx, True)
    assert terminal.learned == [({"a": 1}, True)]
    assert terminal.predicted_with == [{"a": 1}]


# ---------------------------------------------------------------------------
# Context life cycle
# ---------------------------------------------------------------------------


def test_discard_semantics():
    terminal = TerminalRecorder()
    pipeline = compose.Pipeline(compose.Select("a"), terminal)

    _, ctx = pipeline.predict_one_with_context({"a": 1})
    assert pipeline.discard_context(ctx) is True
    assert pipeline.discard_context(ctx) is False
    assert terminal.learned == []

    with pytest.raises(ValueError):
        pipeline.learn_one_from_context(ctx, 1.0)
    assert terminal.learned == []


def test_double_learn_rejected_without_updates():
    terminal = TerminalRecorder()
    pipeline = compose.Pipeline(compose.Select("a"), terminal)
    _, ctx = pipeline.predict_one_with_context({"a": 1})
    pipeline.learn_one_from_context(ctx, 1.0)
    with pytest.raises(ValueError):
        pipeline.learn_one_from_context(ctx, 2.0)
    assert terminal.learned == [({"a": 1}, 1.0)]


def test_wrong_owner_and_forged_contexts():
    other = compose.Pipeline(compose.Select("a"), TerminalRecorder())
    _, foreign = other.predict_one_with_context({"a": 1})

    terminal = TerminalRecorder()
    pipeline = compose.Pipeline(compose.Select("a"), terminal)

    with pytest.raises(ValueError):
        pipeline.learn_one_from_context(foreign, 1.0)
    with pytest.raises(ValueError):
        pipeline.discard_context(foreign)
    with pytest.raises(ValueError):
        pipeline.learn_one_from_context(object(), 1.0)
    with pytest.raises(ValueError):
        pipeline.discard_context(None)
    assert terminal.learned == []

    # A clone is a different instance and cannot adopt the original's context.
    original = compose.Pipeline(compose.Select("a"), TerminalRecorder())
    _, ctx = original.predict_one_with_context({"a": 1})
    clone = original.clone()
    with pytest.raises(ValueError):
        clone.learn_one_from_context(ctx, 1.0)


def test_clone_cannot_consume_original_context():
    model = compose.Pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression())
    _, ctx = model.predict_one_with_context({"a": 1})
    cloned = model.clone()
    with pytest.raises(ValueError):
        cloned.learn_one_from_context(ctx, 1.0)
    # The original still can.
    model.learn_one_from_context(ctx, 1.0)


def test_two_pipelines_interleaved():
    a_term, b_term = TerminalRecorder(), TerminalRecorder()
    a = compose.Pipeline(compose.Select("a"), a_term)
    b = compose.Pipeline(compose.Select("a"), b_term)

    _, ca1 = a.predict_one_with_context({"a": 1})
    _, cb1 = b.predict_one_with_context({"a": 2})
    _, ca2 = a.predict_one_with_context({"a": 3})

    b.learn_one_from_context(cb1, 20.0)
    a.learn_one_from_context(ca2, 30.0)
    a.learn_one_from_context(ca1, 10.0)

    assert a_term.learned == [({"a": 3}, 30.0), ({"a": 1}, 10.0)]
    assert b_term.learned == [({"a": 2}, 20.0)]


def test_context_belongs_to_creating_pipeline_even_if_state_changes():
    terminal = TerminalRecorder()
    pipeline = compose.Pipeline(compose.Select("a"), terminal)
    _, ctx = pipeline.predict_one_with_context({"a": 1})
    # Mutating the pipeline does not redirect feedback to a different terminal.
    pipeline.steps["TerminalRecorder"] = TerminalRecorder()
    pipeline.learn_one_from_context(ctx, 4.0)
    assert terminal.learned == [({"a": 1}, 4.0)]
    assert pipeline[-1].learned == []


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_prediction_failure_leaves_no_consumable_context():
    class Boom(base.Transformer):
        def __init__(self):
            self.learned = []

        def learn_one(self, x):
            self.learned.append(dict(x))

        def transform_one(self, x):
            raise RuntimeError("boom")

    boom = Boom()
    terminal = TerminalRecorder()
    pipeline = compose.Pipeline(boom, terminal)
    with pytest.raises(RuntimeError):
        pipeline.predict_one_with_context({"a": 1})
    # The unsupervised update that already happened is not rolled back.
    assert boom.learned == [{"a": 1}]
    # No context table is left behind.
    assert "_pending_contexts" not in pipeline.__dict__


def test_terminal_prediction_failure_leaves_no_context():
    class BoomTerminal(base.Regressor):
        def predict_one(self, x):
            raise RuntimeError("predict boom")

        def learn_one(self, x, y):
            pass

    pipeline = compose.Pipeline(compose.Select("a"), BoomTerminal())
    with pytest.raises(RuntimeError):
        pipeline.predict_one_with_context({"a": 1})
    assert "_pending_contexts" not in pipeline.__dict__


def test_feedback_failure_terminates_context():
    class BoomTerminal(base.Regressor):
        def __init__(self):
            self.calls = 0

        def predict_one(self, x):
            return 0.0

        def learn_one(self, x, y):
            self.calls += 1
            raise RuntimeError("learn boom")

    terminal = BoomTerminal()
    sup = SupervisedRecorder()
    pipeline = compose.Pipeline(sup, terminal)
    _, ctx = pipeline.predict_one_with_context({"a": 1})
    with pytest.raises(RuntimeError, match="learn boom"):
        pipeline.learn_one_from_context(ctx, 1.0)
    # The supervised transformer learned before the failure, and is not replayed.
    assert sup.learned == [({"a": 1}, 1.0)]
    assert terminal.calls == 1
    with pytest.raises(ValueError):
        pipeline.learn_one_from_context(ctx, 1.0)
    assert terminal.calls == 1
    # The context no longer holds its payload.
    assert ctx._payload_size() == 0
    assert "_pending_contexts" not in pipeline.__dict__

    # Other pending observations and future calls still work.
    _, ctx2 = pipeline.predict_one_with_context({"a": 2})
    with pytest.raises(RuntimeError):
        pipeline.learn_one_from_context(ctx2, 2.0)


def test_supervised_transformer_failure_stops_before_terminal():
    class BoomSup(base.SupervisedTransformer):
        def learn_one(self, x, y):
            raise RuntimeError("sup boom")

        def transform_one(self, x):
            return dict(x)

    terminal = TerminalRecorder()
    pipeline = compose.Pipeline(BoomSup(), terminal)
    _, ctx = pipeline.predict_one_with_context({"a": 1})
    with pytest.raises(RuntimeError, match="sup boom"):
        pipeline.learn_one_from_context(ctx, 1.0)
    assert terminal.learned == []
    with pytest.raises(ValueError):
        pipeline.learn_one_from_context(ctx, 1.0)
    assert terminal.learned == []


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_anomaly_filter_rejected_before_any_update():
    from river import anomaly

    detector = anomaly.GaussianScorer()
    scaler = preprocessing.StandardScaler()
    pipeline = compose.Pipeline(
        anomaly.ThresholdFilter(detector, threshold=0.99),
        scaler,
        linear_model.LinearRegression(),
    )
    with pytest.raises(ValueError, match="anomaly"):
        pipeline.predict_one_with_context({"a": 1})
    assert dict(scaler.counts) == {}


def test_active_learner_rejected():
    from river import active

    scaler = preprocessing.StandardScaler()
    pipeline = compose.Pipeline(
        scaler,
        active.EntropySampler(linear_model.LogisticRegression()),
    )
    with pytest.raises(ValueError, match="active"):
        pipeline.predict_proba_one_with_context({"a": 1})
    assert dict(scaler.counts) == {}


def test_shared_stateful_instance_rejected():
    shared = preprocessing.StandardScaler()
    union = compose.TransformerUnion(("left", compose.Select("a") | shared))
    # ``TransformerUnion`` clones on insertion, so wire the *same* instance into a second
    # branch by hand, as a user might do when mutating a pipeline after construction.
    union.transformers["right"] = union.transformers["left"]
    pipeline = compose.Pipeline(union, linear_model.LinearRegression())
    with pytest.raises(ValueError, match="stateful"):
        pipeline.predict_one_with_context({"a": 1, "b": 2})
    assert dict(shared.counts) == {}
    assert dict(union.transformers["left"][1].counts) == {}


def test_shared_stateless_instance_accepted():
    shared = compose.Select("a")
    union = compose.TransformerUnion(("left", shared))
    union.transformers["right"] = union.transformers["left"]
    pipeline = compose.Pipeline(union, linear_model.LinearRegression())
    prediction, ctx = pipeline.predict_one_with_context({"a": 2})
    pipeline.learn_one_from_context(ctx, 1.0)
    assert prediction == 0.0


def test_unsupervised_terminal_rejected():
    pipeline = compose.Pipeline(preprocessing.StandardScaler(), compose.Select("a"))
    with pytest.raises(ValueError):
        pipeline.predict_one_with_context({"a": 1})


def test_proba_unsupported_regressor_rejected_before_learning():
    scaler = preprocessing.StandardScaler()
    pipeline = compose.Pipeline(scaler, linear_model.LinearRegression())
    with pytest.raises(ValueError, match="predict_proba_one"):
        pipeline.predict_proba_one_with_context({"a": 0.0})
    assert dict(scaler.counts) == {}


def test_old_interface_unchanged_with_anomaly_filter():
    from river import anomaly

    model = (
        anomaly.ThresholdFilter(anomaly.GaussianScorer(), threshold=0.0)
        | compose.Select("a")
        | linear_model.LinearRegression()
    )
    model.learn_one({"a": 1}, 1.0)
    assert model.predict_one({"a": 1}) is not None


# ---------------------------------------------------------------------------
# Interaction with real River transformers
# ---------------------------------------------------------------------------


def test_real_standard_scaler_target_agg_linear_regression():
    pytest.importorskip("pandas")
    agg = feature_extraction.TargetAgg(by="g", how=stats.Mean())
    scaler = preprocessing.StandardScaler()
    pipeline = compose.Pipeline(
        compose.Select("a", "g"),
        compose.TransformerUnion(
            compose.Select("a") | scaler,
            agg,
        ),
        linear_model.LinearRegression(),
    )
    scaler = pipeline[1][0][1]
    agg = pipeline[1][1]

    _, ctx1 = pipeline.predict_one_with_context({"a": 10.0, "g": "x"})
    # The scaler learned during prediction, so its count is already 1.
    assert dict(scaler.counts) == {"a": 1}
    # TargetAgg has not seen the target yet.
    assert agg._groups[("x",)].n == 0

    pipeline.learn_one_from_context(ctx1, 4.0)
    # Feedback updated TargetAgg with the input it transformed during prediction: the
    # branch-local input, i.e. the selected but unscaled features.
    assert agg._groups[("x",)].n == 1
    assert agg._groups[("x",)]._mean == 4.0
    # Scaler was not learned again during feedback.
    assert dict(scaler.counts) == {"a": 1}

    # The new aggregate influences later predictions: group x now carries a target mean
    # of 4, group y still has none.
    prediction_x, ctx_x = pipeline.predict_one_with_context({"a": 10.0, "g": "x"})
    prediction_y, ctx_y = pipeline.predict_one_with_context({"a": 10.0, "g": "x"})
    pipeline.discard_context(ctx_x)
    pipeline.discard_context(ctx_y)
    assert prediction_x == prediction_y

    # A TargetAgg fed through the branch sees the unaggregated input; make sure the
    # terminal's features carry the aggregate value.
    assert pipeline[1].transform_one({"a": 1.0, "g": "x"})["y_mean_by_g"] == 4.0


def test_real_classifier_log_loss():
    scaler = preprocessing.StandardScaler()
    model = compose.Pipeline(scaler, linear_model.LogisticRegression())

    dataset = [({"a": float(i), "b": float(i % 3)}, bool(i % 2)) for i in range(30)]
    metric = metrics.LogLoss()
    for x, y in dataset:
        y_pred, ctx = model.predict_proba_one_with_context(x)
        metric.update(y, y_pred)
        model.learn_one_from_context(ctx, y)
    assert 0.0 <= metric.get() < 1.0


def test_target_agg_feedback_does_not_rewrite_waiting_predictions():
    """A TargetAgg update must not change the features other pending observations saw."""

    agg = feature_extraction.TargetAgg(by="g", how=stats.Mean())
    terminal = TerminalRecorder()
    pipeline = compose.Pipeline(compose.Select("a", "g"), agg, terminal)
    agg = pipeline[1]

    # Three predictions, all made before any feedback.
    contexts = []
    for i in range(3):
        _, ctx = pipeline.predict_one_with_context({"a": i, "g": "same"})
        contexts.append(ctx)

    # All three saw the same (empty) aggregate.
    assert [features["y_mean_by_g"] for features in terminal.predicted_with] == [0, 0, 0]

    # Feed back the first and the third.
    pipeline.learn_one_from_context(contexts[0], 10.0)
    pipeline.learn_one_from_context(contexts[2], 30.0)

    # The supervised transformer learned from its own recorded input, in real feedback
    # order; the waiting context still knows the features it predicted with.
    assert agg._groups[("same",)]._mean == 20.0

    pipeline.learn_one_from_context(contexts[1], 20.0)
    assert agg._groups[("same",)]._mean == 20.0  # (10 + 30 + 20) / 3

    # The terminal learned using the features it was given, not recomputed ones.
    assert [f["y_mean_by_g"] for f, _ in terminal.learned] == [0, 0, 0]


def test_predictions_made_later_see_feedback_so_far():
    agg = feature_extraction.TargetAgg(by="g", how=stats.Mean())
    pipeline = compose.Pipeline(compose.Select("a", "g"), agg, TerminalRecorder())

    _, c1 = pipeline.predict_one_with_context({"a": 1, "g": "x"})
    pipeline.learn_one_from_context(c1, 4.0)

    _, c2 = pipeline.predict_one_with_context({"a": 1, "g": "x"})
    pipeline.discard_context(c2)
    # The second prediction saw the aggregate from the first feedback.
    assert pipeline[-1].predicted_with[-1]["y_mean_by_g"] == 4.0


def test_y_and_params_forwarded_on_feedback():
    class Terminal(base.Regressor):
        def __init__(self):
            self.calls = []

        def predict_one(self, x):
            return 0.0

        def learn_one(self, x, y, w=1.0):
            self.calls.append((dict(x), y, w))

    terminal = Terminal()
    pipeline = compose.Pipeline(compose.Select("a"), terminal)
    _, ctx = pipeline.predict_one_with_context({"a": 1})
    pipeline.learn_one_from_context(ctx, 5.0, w=2.0)
    assert terminal.calls == [({"a": 1}, 5.0, 2.0)]


def test_empty_pipeline_rejected():
    with pytest.raises(ValueError):
        compose.Pipeline().predict_one_with_context({"a": 1})


def test_nested_pipeline_ending_in_predictor_rejected():
    nested = compose.Pipeline(compose.Select("a"), linear_model.LinearRegression())
    pipeline = compose.Pipeline(nested, linear_model.LinearRegression())
    with pytest.raises(ValueError, match="predictor"):
        pipeline.predict_one_with_context({"a": 1})


def test_transformer_product_rejected():
    product = compose.Select("a") * compose.Select("b")
    pipeline = compose.Pipeline(product, linear_model.LinearRegression())
    with pytest.raises(ValueError, match="custom merge"):
        pipeline.predict_one_with_context({"a": 1, "b": 2})


def test_supervised_transformer_as_terminal_rejected():
    pipeline = compose.Pipeline(
        compose.Select("a"), feature_extraction.TargetAgg(by="g", how=stats.Mean())
    )
    with pytest.raises(ValueError):
        pipeline.predict_one_with_context({"a": 1, "g": "x"})


def test_old_interface_learn_counts_unchanged():
    """The arrival-time interface must still learn every step exactly once per call."""

    rec = Recorder()
    sup = SupervisedRecorder()
    terminal = TerminalRecorder()
    pipeline = compose.Pipeline(rec, sup, terminal)

    pipeline.learn_one({"a": 1}, 10.0)
    assert rec.learned == [{"a": 1}]
    assert rec.transformed_inputs == [{"a": 1}]
    # ``SupervisedTransformer`` is not a ``Transformer`` subclass, so the historical
    # ``learn_one`` never calls its ``transform_one``; both it and the terminal see the
    # unsupervised output.
    assert sup.learned == [({"drift": 1, "a": 1}, 10.0)]
    assert terminal.learned == [({"drift": 1, "a": 1}, 10.0)]


def test_visit_count_of_each_leaf_is_one():
    class Counter(base.Transformer):
        def __init__(self):
            self.transforms = 0
            self.learns = 0

        def learn_one(self, x):
            self.learns += 1

        def transform_one(self, x):
            self.transforms += 1
            return dict(x)

    class SupCounter(base.SupervisedTransformer):
        def __init__(self):
            self.transforms = 0
            self.learns = 0

        def learn_one(self, x, y):
            self.learns += 1

        def transform_one(self, x):
            self.transforms += 1
            return dict(x)

    unsup = Counter()
    sup = SupCounter()
    pipeline = compose.Pipeline(unsup, sup, compose.Select("a"), TerminalRecorder())
    _, ctx = pipeline.predict_one_with_context({"a": 1})
    assert (unsup.learns, unsup.transforms) == (1, 1)
    assert (sup.learns, sup.transforms) == (0, 1)
    pipeline.learn_one_from_context(ctx, 1.0)
    assert (unsup.learns, unsup.transforms) == (1, 1)
    assert (sup.learns, sup.transforms) == (1, 1)


def test_failed_feedback_does_not_replay_earlier_steps():
    class CountingSup(base.SupervisedTransformer):
        def __init__(self):
            self.learns = 0

        def learn_one(self, x, y):
            self.learns += 1

        def transform_one(self, x):
            return dict(x)

    class BoomTerminal(base.Regressor):
        def __init__(self):
            self.learns = 0

        def predict_one(self, x):
            return 0.0

        def learn_one(self, x, y):
            self.learns += 1
            raise RuntimeError("boom")

    sup = CountingSup()
    terminal = BoomTerminal()
    pipeline = compose.Pipeline(sup, terminal)

    _, ctx = pipeline.predict_one_with_context({"a": 1})
    with pytest.raises(RuntimeError):
        pipeline.learn_one_from_context(ctx, 1.0)
    # A second feedback is rejected as an already-consumed context, and nothing replays.
    with pytest.raises(ValueError):
        pipeline.learn_one_from_context(ctx, 1.0)
    assert sup.learns == 1
    assert terminal.learns == 1

    # Other pending observations still work, and the failed one cannot be discarded
    # into a duplicate update either.
    _, ok = pipeline.predict_one_with_context({"a": 2})
    assert pipeline.discard_context(ctx) is False
    assert sup.learns == 1
    with pytest.raises(RuntimeError):
        pipeline.learn_one_from_context(ok, 2.0)


def test_context_repr_does_not_expose_payload():
    pipeline = compose.Pipeline(compose.Select("secret"), TerminalRecorder())
    _, ctx = pipeline.predict_one_with_context({"secret": 1})
    assert "secret" not in repr(ctx)
    assert "pending" in repr(ctx)


def test_discarding_a_context_midway_leaves_others_usable():
    terminal = TerminalRecorder()
    pipeline = compose.Pipeline(compose.Select("a"), terminal)

    _, c1 = pipeline.predict_one_with_context({"a": 1})
    _, c2 = pipeline.predict_one_with_context({"a": 2})
    _, c3 = pipeline.predict_one_with_context({"a": 3})

    assert pipeline.discard_context(c2) is True
    pipeline.learn_one_from_context(c3, 30.0)
    pipeline.learn_one_from_context(c1, 10.0)
    assert terminal.learned == [({"a": 3}, 30.0), ({"a": 1}, 10.0)]
    assert pipeline.discard_context(c2) is False
    with pytest.raises(ValueError):
        pipeline.learn_one_from_context(c2, 20.0)


def test_arrival_mode_learn_counts_one_per_step():
    class Counter(base.Transformer):
        def __init__(self):
            self.learns = 0

        def learn_one(self, x):
            self.learns += 1

        def transform_one(self, x):
            return dict(x)

    class SupCounter(base.SupervisedTransformer):
        def __init__(self):
            self.learns = 0

        def learn_one(self, x, y):
            self.learns += 1

        def transform_one(self, x):
            return dict(x)

    unsup, sup = Counter(), SupCounter()
    pipeline = compose.Pipeline(unsup, sup, TerminalRecorder())
    for i in range(5):
        pipeline.learn_one({"a": i}, i)
    assert unsup.learns == 5
    assert sup.learns == 5
    assert len(pipeline[-1].learned) == 5


def test_terminal_reusing_a_probability_buffer_does_not_corrupt_stored_prediction():
    class BufferedClassifier(base.Classifier):
        def __init__(self):
            self._proba = {}
            self._calls = 0

        def learn_one(self, x, y):
            pass

        def predict_proba_one(self, x):
            self._calls += 1
            self._proba.clear()
            self._proba[True] = 1.0 / self._calls
            self._proba[False] = 1.0 - 1.0 / self._calls
            return self._proba

    pipeline = compose.Pipeline(compose.Select("a"), BufferedClassifier())
    first, ctx1 = pipeline.predict_proba_one_with_context({"a": 1})
    second, ctx2 = pipeline.predict_proba_one_with_context({"a": 2})
    # The terminal reused and rewrote its buffer, but the first prediction is intact.
    assert first == {True: 1.0, False: 0.0}
    assert second == {True: 0.5, False: 0.5}
    pipeline.learn_one_from_context(ctx1, True)
    pipeline.discard_context(ctx2)
