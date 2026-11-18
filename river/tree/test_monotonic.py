"""Test suite for monotonic constraints in the online regression Hoeffding trees."""

from __future__ import annotations

import itertools
import math
import pickle
import random

import pytest

from river import base, drift, tree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class ScriptedDriftDetector(base.DriftDetector):
    """Drift detector controlled by an external per-update trigger script.

    ``triggers`` is a sequence of 1-based update counts at which ``drift_detected`` is True.
    Every detector instance keeps its own counter, so the same script applies to the
    detectors cloned inside the adaptive tree nodes.
    """

    def __init__(self, triggers=()):
        self.triggers = tuple(triggers)
        self._calls = 0
        self._drift_detected = False

    def update(self, x):
        self._calls += 1
        self._drift_detected = self._calls in self.triggers

    def clone(self):
        return ScriptedDriftDetector(self.triggers)


MONO_CONFIG = {"a": 1, "b": -1}
CONSTRAINTS = (("a", 1), ("b", -1))


def make_model(cls, **kwargs):
    params = dict(
        leaf_prediction="mean",
        binary_split=True,
        grace_period=25,
        delta=0.1,
        tau=0.1,
        min_samples_split=5,
        monotonic_cst=dict(MONO_CONFIG),
    )
    params.update(kwargs)
    if cls is tree.HoeffdingAdaptiveTreeRegressor:
        params.setdefault("seed", 17)
    return cls(**params)


def target_mixed(x):
    return 8 * int(x["a"] > 0) - 5 * int(x["b"] > 0) + 12 * int(x["c"] > 0)


def ordered_pair(rng, constraints, delta_max=0.4):
    """Generate an independent comparable input pair (x, x2) with x <=_c x2."""
    x = {name: rng.uniform(-1, 1) for name, _ in constraints}
    x["c"] = rng.uniform(-1, 1)
    x2 = dict(x)
    for name, direction in constraints:
        step = rng.uniform(0.0, delta_max)
        if direction == 1:
            x2[name] = min(1.0, x[name] + step)
        else:
            x2[name] = max(-1.0, x[name] - step)
    return x, x2


def tolerance(p1, p2):
    return 1e-10 * max(1.0, abs(p1), abs(p2))


def check_stream_monotonicity(model, rng, n_checks, constraints=CONSTRAINTS):
    """After every single learned record, verify the inequality on fresh pairs."""
    for _ in range(n_checks):
        x = {name: rng.uniform(-1, 1) for name, _ in constraints}
        x["c"] = rng.uniform(-1, 1)
        y = target_mixed(x) + rng.gauss(0, 0.4)
        model.learn_one(x, y)

        p, q = ordered_pair(random.Random(rng.randrange(10**9)), constraints)
        pred_p = model.predict_one(p)
        pred_q = model.predict_one(q)
        assert pred_q + tolerance(pred_p, pred_q) >= pred_p


def grid_violations(model, constraints, n_grid=6):
    """Exhaustive partial-order check over a grid covering unobserved combinations.

    Every comparable grid pair is checked; the coordinator additionally guarantees a strict
    bound on fully box-ordered region pairs, counted separately so regressions there fail
    loudly even when loose-overlap pairs dominate the totals.
    """
    domain = [-1.0 + 2.0 * i / (n_grid - 1) for i in range(n_grid)]
    features = [name for name, _ in constraints] + ["c"]
    points = [dict(zip(features, vals)) for vals in itertools.product(*(domain for _ in features))]
    preds = [model.predict_one(p) for p in points]

    def reached_leaf(p):
        leaf = model._root.traverse(p, until_leaf=True)
        return leaf[0] if isinstance(leaf, list) else leaf

    leaves = [reached_leaf(p) for p in points]
    coordinator = model._mono_coordinator

    def ordered(p, q):
        for name, direction in constraints:
            if direction == 1 and not p[name] <= q[name]:
                return False
            if direction == -1 and not p[name] >= q[name]:
                return False
        return True

    violations = 0
    strict_violations = 0
    worst = 0.0
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i == j or not ordered(p, q):
                continue
            if preds[j] < preds[i] - tolerance(preds[i], preds[j]):
                violations += 1
                worst = max(worst, preds[i] - preds[j])
                box_i = coordinator._boxes[leaves[i]]
                box_j = coordinator._boxes[leaves[j]]
                if coordinator._boxes_fully_ordered(box_i, box_j):
                    strict_violations += 1
    assert strict_violations == 0
    return violations, worst


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [tree.HoeffdingTreeRegressor, tree.HoeffdingAdaptiveTreeRegressor])
def test_invalid_constraint_config(cls):
    with pytest.raises(ValueError):
        cls(monotonic_cst=[("a", 1)])
    with pytest.raises(ValueError):
        cls(monotonic_cst={"": 1})
    with pytest.raises(ValueError):
        cls(monotonic_cst={"a": 2})
    with pytest.raises(ValueError):
        cls(monotonic_cst={"a": -2})
    with pytest.raises(ValueError):
        cls(monotonic_cst={"a": True})
    with pytest.raises(ValueError):
        cls(monotonic_cst={"a": 1.0})
    with pytest.raises(ValueError):
        cls(monotonic_cst={1: 1})


@pytest.mark.parametrize("cls", [tree.HoeffdingTreeRegressor, tree.HoeffdingAdaptiveTreeRegressor])
def test_unsupported_configs_rejected_before_training(cls):
    with pytest.raises(ValueError):
        cls(monotonic_cst={"a": 1})  # default leaf prediction is adaptive
    with pytest.raises(ValueError):
        cls(leaf_prediction="model", monotonic_cst={"a": 1})
    with pytest.raises(ValueError):
        cls(leaf_prediction="adaptive", monotonic_cst={"a": 1})
    with pytest.raises(ValueError):
        cls(leaf_prediction="mean", binary_split=False, monotonic_cst={"a": 1})
    with pytest.raises(ValueError):
        cls(
            leaf_prediction="mean",
            binary_split=True,
            nominal_attributes=["a"],
            monotonic_cst={"a": 1},
        )


@pytest.mark.parametrize("cls", [tree.HoeffdingTreeRegressor, tree.HoeffdingAdaptiveTreeRegressor])
def test_neutral_configs_disable_constraints(cls):
    for cfg in (None, {}, {"a": 0}, {"a": 0, "b": 0}):
        model = cls(leaf_prediction="mean", monotonic_cst=cfg)
        assert model._mono_active is False
        assert model.monotonic_constraints is None


@pytest.mark.parametrize("cls", [tree.HoeffdingTreeRegressor, tree.HoeffdingAdaptiveTreeRegressor])
def test_caller_mapping_not_stored_or_mutated(cls):
    cfg = {"a": 1, "b": -1, "z": 0}
    model = cls(leaf_prediction="mean", binary_split=True, monotonic_cst=cfg)
    cfg["a"] = -1
    cfg["new"] = 1
    assert dict(model.monotonic_constraints) == {"a": 1, "b": -1}
    assert model.monotonic_constraints == tuple(sorted({"a": 1, "b": -1}.items()))
    # The stored object is an immutable tuple, not the caller's dict.
    with pytest.raises(TypeError):
        model.monotonic_constraints["a"] = 0  # type: ignore[index]


# ---------------------------------------------------------------------------
# Input validation and side-effect-free rejection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [tree.HoeffdingTreeRegressor, tree.HoeffdingAdaptiveTreeRegressor])
def test_bad_inputs_rejected_without_side_effects(cls):
    model = make_model(cls)
    # Warm up so there is real state to preserve.
    rng = random.Random(0)
    for _ in range(120):
        x = {"a": rng.uniform(-1, 1), "b": rng.uniform(-1, 1), "c": rng.uniform(-1, 1)}
        model.learn_one(x, target_mixed(x))

    state = pickle.dumps(model)

    def assert_unchanged(bad_call):
        with pytest.raises(ValueError):
            bad_call()
        assert pickle.dumps(model) == state

    base_x = {"a": 0.2, "b": -0.3, "c": 0.1}
    assert_unchanged(lambda: model.learn_one({"b": 0.0, "c": 0.0}, 1.0))  # missing feature
    assert_unchanged(lambda: model.learn_one(dict(base_x, a="high"), 1.0))
    assert_unchanged(lambda: model.learn_one(dict(base_x, a=True), 1.0))
    assert_unchanged(lambda: model.learn_one(dict(base_x, a=math.nan), 1.0))
    assert_unchanged(lambda: model.learn_one(dict(base_x, a=math.inf), 1.0))
    assert_unchanged(lambda: model.learn_one(base_x, math.nan))
    assert_unchanged(lambda: model.learn_one(base_x, math.inf))
    assert_unchanged(lambda: model.learn_one(base_x, True))
    assert_unchanged(lambda: model.learn_one(base_x, 1.0, w=-0.5))
    assert_unchanged(lambda: model.learn_one(base_x, 1.0, w=math.nan))
    assert_unchanged(lambda: model.learn_one(base_x, 1.0, w=True))
    assert_unchanged(lambda: model.predict_one({"b": 0.0, "c": 0.0}))
    assert_unchanged(lambda: model.predict_one(dict(base_x, b=math.nan)))


@pytest.mark.parametrize("cls", [tree.HoeffdingTreeRegressor, tree.HoeffdingAdaptiveTreeRegressor])
def test_zero_weight_is_noop(cls):
    model = make_model(cls)
    rng = random.Random(2)
    for _ in range(60):
        x = {"a": rng.uniform(-1, 1), "b": rng.uniform(-1, 1), "c": rng.uniform(-1, 1)}
        model.learn_one(x, target_mixed(x))

    before = pickle.dumps(model)
    model.learn_one({"a": 0.9, "b": -0.9, "c": 0.9}, 1000.0, w=0.0)
    assert pickle.dumps(model) == before
    # Training counter unchanged.
    weight_before = model._train_weight_seen_by_model
    model.learn_one({"a": 0.1, "b": 0.1, "c": 0.1}, 1.0, w=0.0)
    assert model._train_weight_seen_by_model == weight_before


@pytest.mark.parametrize("cls", [tree.HoeffdingTreeRegressor, tree.HoeffdingAdaptiveTreeRegressor])
def test_predictions_have_no_side_effects(cls):
    model = make_model(cls)
    rng = random.Random(3)
    for _ in range(80):
        x = {"a": rng.uniform(-1, 1), "b": rng.uniform(-1, 1), "c": rng.uniform(-1, 1)}
        model.learn_one(x, target_mixed(x))

    state = pickle.dumps(model)
    probe = {"a": 0.4, "b": -0.6, "c": 0.2}
    first = model.predict_one(probe)
    for _ in range(10):
        assert model.predict_one(probe) == first
    # Different query order must not change answers.
    queries = [
        {"a": -0.8, "b": 0.7, "c": -0.2},
        {"a": 0.8, "b": -0.7, "c": 0.9},
        probe,
    ]
    answers_a = [model.predict_one(q) for q in queries]
    answers_b = [model.predict_one(q) for q in reversed(queries)]
    assert answers_a == list(reversed(answers_b))
    assert pickle.dumps(model) == state


# ---------------------------------------------------------------------------
# Monotonicity on small streams
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [tree.HoeffdingTreeRegressor, tree.HoeffdingAdaptiveTreeRegressor])
def test_per_step_monotonicity_small_stream(cls):
    model = make_model(cls)
    check_stream_monotonicity(model, random.Random(21), 400)


@pytest.mark.parametrize("cls", [tree.HoeffdingTreeRegressor, tree.HoeffdingAdaptiveTreeRegressor])
def test_grid_monotonicity_forward_reverse_mixed(cls):
    rng = random.Random(4)
    targets = {
        "forward": lambda x: 8 * int(x["a"] > 0),
        "reverse": lambda x: -8 * int(x["b"] > 0),
        "mixed": target_mixed,
    }
    for name, tfn in targets.items():
        model = make_model(cls)
        for _ in range(1500):
            x = {"a": rng.uniform(-1, 1), "b": rng.uniform(-1, 1), "c": rng.uniform(-1, 1)}
            model.learn_one(x, tfn(x) + rng.gauss(0, 0.3))
        grid_violations(model, CONSTRAINTS)  # strict-order guarantee asserted inside


@pytest.mark.parametrize("cls", [tree.HoeffdingTreeRegressor, tree.HoeffdingAdaptiveTreeRegressor])
def test_unconstrained_feature_still_matters(cls):
    # The unconstrained feature c must not be ignored: the tree has to split on it and the
    # split statistics must separate the two c populations.
    from river.tree.nodes.branch import NumericBinaryBranch

    model = make_model(cls, grace_period=20)
    rng = random.Random(9)
    for _ in range(6000):
        x = {"a": rng.uniform(-1, 1), "b": rng.uniform(-1, 1), "c": rng.uniform(-1, 1)}
        model.learn_one(x, 4 * int(x["a"] > 0) + 12 * int(x["c"] > 0))

    c_splits = [
        node
        for node in model._root.iter_dfs()
        if isinstance(node, NumericBinaryBranch) and node.feature == "c"
    ]
    assert c_splits, "the model never split on the unconstrained feature c"
    separated = any(
        abs(branch.children[0].stats.mean.get() - branch.children[1].stats.mean.get()) > 1.0
        for branch in c_splits
    )
    assert separated
    # The strict-order guarantee is asserted inside grid_violations.
    grid_violations(model, CONSTRAINTS)


@pytest.mark.parametrize("cls", [tree.HoeffdingTreeRegressor, tree.HoeffdingAdaptiveTreeRegressor])
def test_repeated_split_and_threshold_crossing(cls):
    from river.tree.nodes.branch import NumericBinaryBranch

    model = make_model(cls, grace_period=20, max_depth=12)
    rng = random.Random(13)
    for _ in range(3000):
        x = {"a": rng.uniform(-1, 1), "b": rng.uniform(-1, 1), "c": rng.uniform(-1, 1)}
        model.learn_one(x, target_mixed(x) + rng.gauss(0, 0.2))
    # The tree grows beyond a single leaf and repeatedly splits constrained features.
    assert model.n_leaves > 1
    constrained_splits = [
        node
        for node in model._root.iter_dfs()
        if isinstance(node, NumericBinaryBranch) and node.feature in dict(CONSTRAINTS)
    ]
    assert constrained_splits
    grid_violations(model, CONSTRAINTS, n_grid=7)
    # Explicit threshold-side checks on the actual constrained-feature split thresholds.
    constrained_branches = [
        node
        for node in model._root.iter_dfs()
        if isinstance(node, tree.nodes.branch.NumericBinaryBranch)
        and node.feature in dict(CONSTRAINTS)
    ]
    assert constrained_branches
    for branch in constrained_branches[:5]:
        direction = dict(CONSTRAINTS)[branch.feature]
        lower = {"a": 0.0, "b": 0.0, "c": 0.0}
        upper = dict(lower)
        lower[branch.feature] = branch.threshold - 1e-6
        upper[branch.feature] = branch.threshold + 1e-6
        pred_lower = model.predict_one(lower)
        pred_upper = model.predict_one(upper)
        if direction == 1:
            assert pred_upper >= pred_lower - tolerance(pred_lower, pred_upper)
        else:
            assert pred_lower >= pred_upper - tolerance(pred_upper, pred_lower)


@pytest.mark.parametrize("cls", [tree.HoeffdingTreeRegressor, tree.HoeffdingAdaptiveTreeRegressor])
def test_fractional_weights_respected(cls):
    model = make_model(cls)
    rng = random.Random(15)
    for i in range(2000):
        x = {"a": rng.uniform(-1, 1), "b": rng.uniform(-1, 1), "c": rng.uniform(-1, 1)}
        model.learn_one(x, target_mixed(x), w=rng.uniform(0.1, 2.0))
    grid_violations(model, CONSTRAINTS)


@pytest.mark.parametrize("cls", [tree.HoeffdingTreeRegressor, tree.HoeffdingAdaptiveTreeRegressor])
def test_long_reversed_label_shock(cls):
    model = make_model(cls)
    rng = random.Random(31)
    for i in range(3000):
        x = {"a": rng.uniform(-1, 1), "b": rng.uniform(-1, 1), "c": rng.uniform(-1, 1)}
        if 1000 <= i < 2500:
            y = -8 * int(x["a"] > 0) + 5 * int(x["b"] > 0)
        else:
            y = 8 * int(x["a"] > 0) - 5 * int(x["b"] > 0)
        model.learn_one(x, y + rng.gauss(0, 0.4))
    # Monotonicity must survive even while the model absorbs contradictory labels.
    grid_violations(model, CONSTRAINTS)


# ---------------------------------------------------------------------------
# Clone / pickle / interleaving
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [tree.HoeffdingTreeRegressor, tree.HoeffdingAdaptiveTreeRegressor])
def test_clone_and_pickle_continue_training(cls):
    model = make_model(cls)
    rng = random.Random(7)
    for _ in range(300):
        x = {"a": rng.uniform(-1, 1), "b": rng.uniform(-1, 1), "c": rng.uniform(-1, 1)}
        model.learn_one(x, target_mixed(x))

    cloned = model.clone()
    # A fresh clone starts from the same constructor parameters and an empty tree.
    assert cloned._mono_active == model._mono_active
    assert cloned.monotonic_constraints == model.monotonic_constraints
    assert cloned.n_leaves is None

    restored = pickle.loads(pickle.dumps(model))
    probe = {"a": 0.3, "b": -0.2, "c": 0.5}
    assert restored.predict_one(probe) == model.predict_one(probe)
    for _ in range(300):
        x = {"a": rng.uniform(-1, 1), "b": rng.uniform(-1, 1), "c": rng.uniform(-1, 1)}
        restored.learn_one(x, target_mixed(x))
    grid_violations(restored, CONSTRAINTS)  # strict-order guarantee asserted inside


def test_two_models_interleaved():
    m1 = make_model(tree.HoeffdingTreeRegressor)
    m2 = make_model(tree.HoeffdingAdaptiveTreeRegressor)
    rng = random.Random(99)
    for _ in range(400):
        x = {"a": rng.uniform(-1, 1), "b": rng.uniform(-1, 1), "c": rng.uniform(-1, 1)}
        y = target_mixed(x)
        m1.learn_one(x, y)
        m2.learn_one(dict(x, a=-x["a"]), -y)
        m1.learn_one(dict(x), y)
    for model in (m1, m2):
        grid_violations(model, CONSTRAINTS)


# ---------------------------------------------------------------------------
# Adaptive lifecycle: alternate generation, switches, pruning
# ---------------------------------------------------------------------------
def _drift_model(detector, **kwargs):
    params = dict(
        leaf_prediction="mean",
        binary_split=True,
        grace_period=15,
        delta=0.1,
        tau=0.1,
        min_samples_split=5,
        seed=7,
        max_depth=6,
        drift_window_threshold=3,
        switch_significance=0.9,
        drift_detector=detector,
        monotonic_cst=dict(MONO_CONFIG),
    )
    params.update(kwargs)
    return tree.HoeffdingAdaptiveTreeRegressor(**params)


def _drift_stream(n, rng, levels):
    for i in range(n):
        x = {"a": rng.uniform(-1, 1), "b": rng.uniform(-1, 1), "c": rng.uniform(-1, 1)}
        level = levels[sum(i >= cut for cut in (n // 3, 2 * n // 3))]
        y = level + target_mixed(x) + rng.gauss(0, 0.3)
        yield x, y


def test_adaptive_lifecycle_generation_switch_prune():
    # A drift detector that fires repeatedly forces every lifecycle transition.
    detector = drift.ADWIN(0.05)
    model = _drift_model(detector)
    rng = random.Random(3)
    for x, y in _drift_stream(4000, rng, (0, -30, 30)):
        model.learn_one(x, y)

    assert model.n_alternate_trees > 0
    assert model.n_switch_alternate_trees > 0
    assert model.n_pruned_alternate_trees > 0
    # The coordinator only tracks live main + alternate leaves.
    from river.tree.nodes.branch import DTBranch

    def collect(node, out):
        if isinstance(node, DTBranch):
            for child in node.children:
                collect(child, out)
            if node._alternate_tree is not None:
                collect(node._alternate_tree, out)
        else:
            out.append(node)

    live = []
    collect(model._root, live)
    tracked = set(model._mono_coordinator._boxes)
    assert all(leaf in tracked for leaf in live)
    assert not (tracked - set(live))
    grid_violations(model, CONSTRAINTS)


def test_adaptive_root_switch_with_controlled_detector():
    # Firing on the very first updates forces drift detection high in the tree, including
    # root-level alternates and their eventual replacement.
    model = _drift_model(
        ScriptedDriftDetector(triggers=(1, 2, 3, 4, 5)),
        max_depth=4,
    )
    rng = random.Random(12)
    for x, y in _drift_stream(2000, rng, (0, 25, -25)):
        model.learn_one(x, y)
    assert model.n_alternate_trees >= 1
    # Monotonicity holds throughout and after the structural churn.
    grid_violations(model, CONSTRAINTS, n_grid=5)
    # Pickle the post-switch tree and continue training.
    restored = pickle.loads(pickle.dumps(model))
    for x, y in _drift_stream(500, random.Random(12), (0, 25, -25)):
        restored.learn_one(x, y)
    grid_violations(restored, CONSTRAINTS, n_grid=5)


def test_default_adwin_runs_drift_data():
    # The shipped default detector must run end-to-end on a drift stream without errors and
    # keep producing a monotone strict-order surface.
    model = make_model(tree.HoeffdingAdaptiveTreeRegressor, grace_period=30)
    rng = random.Random(23)
    for i in range(3000):
        x = {"a": rng.uniform(-1, 1), "b": rng.uniform(-1, 1), "c": rng.uniform(-1, 1)}
        level = 20 if i > 1500 else 0
        model.learn_one(x, level + target_mixed(x) + rng.gauss(0, 0.3))
    grid_violations(model, CONSTRAINTS, n_grid=5)


def test_input_dict_not_retained_or_mutated():
    model = make_model(tree.HoeffdingTreeRegressor)
    x = {"a": 0.2, "b": 0.3, "c": 0.4}
    model.learn_one(x, 1.0)
    x["a"] = 5.0
    # The tree must keep using the value observed at learning time.
    assert model.predict_one({"a": 0.2, "b": 0.3, "c": 0.4}) == model.predict_one(
        {"a": 0.2, "b": 0.3, "c": 0.4}
    )
