from __future__ import annotations

import collections
import contextlib
import functools
import inspect
import io
import itertools
import types
import typing
import weakref

import pandas as pd

from river import base, utils

from . import func, union
from .context import _PredictionContext

__all__ = ["Pipeline"]


# ---------------------------------------------------------------------------
# Prediction-time feature snapshots
# ---------------------------------------------------------------------------
#
# The functions below implement the two-phase protocol used by
# ``predict_*_with_context``. They are deliberately independent from the historical
# ``learn_one`` / ``_transform_one`` code paths, so that the arrival-time semantics of the
# old interface are left untouched.


def _is_transformer_like(obj) -> bool:
    """Whether ``obj`` follows the single-instance transformer protocol.

    River's built-in transformers inherit from ``base.Transformer``. We also accept
    structurally compatible objects so that user-defined steps do not have to inherit from
    a specific class. What matters here is the ``learn_one`` / ``transform_one`` protocol,
    not a class name.

    """

    if isinstance(obj, (base.Transformer, Pipeline, union.TransformerUnion)):
        return True
    return callable(getattr(obj, "transform_one", None)) and callable(
        getattr(obj, "learn_one", None)
    )


def _is_supervised_step(obj) -> bool:
    supervised = getattr(obj, "_supervised", None)
    if isinstance(supervised, bool):
        return supervised
    if supervised is not None:
        return bool(supervised)

    # Duck-typed transformers may not expose ``_supervised``. In that case, look at the
    # signature of learn_one: an extra required positional argument means a target.
    try:
        signature = inspect.signature(obj.learn_one)
    except (TypeError, ValueError):
        return False
    required = [
        p
        for p in signature.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    ]
    return len(required) > 1


_PARAM_NAMES_CACHE: dict[type, frozenset | None] = {}
_LEARN_KWARGS_CACHE: dict[type, tuple[bool, frozenset]] = {}


def _learn_kwargs_support(obj) -> tuple[bool, frozenset]:
    """Whether ``obj.learn_one`` accepts arbitrary kwargs, and which named ones it takes."""

    klass = type(obj)
    cached = _LEARN_KWARGS_CACHE.get(klass)
    if cached is not None:
        return cached
    try:
        parameters = inspect.signature(obj.learn_one).parameters
    except (TypeError, ValueError):
        cached = (False, frozenset())
    else:
        accepts_var_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
        names = frozenset(
            name
            for name, p in parameters.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )
        cached = (accepts_var_kwargs, names)
    _LEARN_KWARGS_CACHE[klass] = cached
    return cached


def _filter_learn_kwargs(obj, params: dict) -> dict:
    if not params:
        return {}
    accepts_var_kwargs, names = _learn_kwargs_support(obj)
    if accepts_var_kwargs:
        return params
    return {k: v for k, v in params.items() if k in names}


def _constructor_param_names(obj) -> frozenset | None:
    """Return the constructor's parameter names for ``type(obj)``, cached per class.

    ``None`` means the signature could not be inspected; callers treat that as "assume
    stateful", which is the safe answer.
    """

    klass = type(obj)
    if klass in _PARAM_NAMES_CACHE:
        return _PARAM_NAMES_CACHE[klass]
    try:
        parameters = inspect.signature(obj.__init__).parameters  # type: ignore[misc]
    except (TypeError, ValueError):
        _PARAM_NAMES_CACHE[klass] = None
        return None
    names = frozenset(
        name
        for name, param in parameters.items()
        if name != "self" and param.kind != inspect.Parameter.VAR_KEYWORD
    )
    _PARAM_NAMES_CACHE[klass] = names
    return names


_SLOT_NAMES_CACHE: dict[type, frozenset] = {}


def _declared_slot_names(klass) -> frozenset:
    cached = _SLOT_NAMES_CACHE.get(klass)
    if cached is not None:
        return cached
    names = set()
    for parent in klass.__mro__:
        slots = parent.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            names.add(slots)
        else:
            names.update(slots)
    cached = frozenset(name for name in names if not name.startswith("__"))
    _SLOT_NAMES_CACHE[klass] = cached
    return cached


def _is_stateful_step(obj) -> bool:
    """Best-effort detection of a transformer that keeps state beyond its parameters.

    Sharing a *stateless* instance between branches is harmless: it applies the same pure
    function twice. Sharing a *stateful* instance is not supported by the two-phase
    protocol, because two branches would record and update the same state.

    The slot names and constructor parameters only depend on the class and are cached;
    only the instance's current attribute names are inspected per call.
    """

    state = getattr(obj, "__dict__", None)
    slot_names = _declared_slot_names(type(obj))

    if not state and not slot_names:
        return False

    attributes = set(state or ())
    attributes.update(slot_names)
    if not attributes:
        return False

    allowed = _constructor_param_names(obj)
    if allowed is None:
        return True
    # Attributes that are not constructor parameters are learned state.
    return any(attribute not in allowed for attribute in attributes)


def _clone_prediction(proba: dict) -> dict:
    """Protect a prediction dictionary from later mutation by the caller."""
    return dict(proba)


class _PendingContexts:
    """Bookkeeping for the contexts a pipeline is still waiting to feed back.

    Contexts are held through weak references, so a caller that simply drops a context
    does not leak it. The container also detaches itself from its pipeline as soon as it
    empties, which means a pipeline with nothing pending keeps no context table at all
    and never accumulates the identities of finished observations.
    """

    __slots__ = ("_owner_ref", "_owner_attr", "_refs")

    def __init__(self, owner, owner_attr):
        self._bind(owner, owner_attr)

    def add(self, context):
        key = id(context)

        def _forget(dead_ref, key=key):
            # Only remove the entry if it is still ours: the id might have been reused by
            # a later context registered under the same key.
            if self._refs.get(key) is dead_ref:
                self._refs.pop(key, None)
                self._maybe_detach()

        self._refs[key] = weakref.ref(context, _forget)

    def discard(self, context):
        key = id(context)
        ref = self._refs.get(key)
        if ref is None:
            return
        if ref() is not context:
            # ``id`` was reused by another live context; leave it alone.
            return
        self._refs.pop(key, None)
        self._maybe_detach()

    def _maybe_detach(self):
        if self._refs:
            return
        owner = self._owner_ref()
        if owner is not None and owner.__dict__.get(self._owner_attr) is self:
            owner.__dict__.pop(self._owner_attr, None)

    def __len__(self):
        return len(self._refs)

    def __bool__(self):
        return bool(self._refs)

    def __iter__(self):
        for ref in list(self._refs.values()):
            context = ref()
            if context is not None:
                yield context

    def __deepcopy__(self, memo):
        # A cloned pipeline must not inherit another pipeline's pending observations.
        # Produce a detached, empty container. It is rebound to the clone the first time
        # that clone registers a context.
        clone = _PendingContexts.__new__(_PendingContexts)
        clone._refs = {}
        clone._owner_attr = "_pending_contexts"
        clone._owner_ref = lambda: None
        return clone

    def _is_bound_to(self, owner) -> bool:
        return self._owner_ref() is owner

    def _bind(self, owner, owner_attr):
        self._owner_ref = weakref.ref(owner)
        self._owner_attr = owner_attr
        self._refs = {}


def _has_builtin_union_semantics(obj) -> bool:
    """Whether a union merges branch outputs the standard ``TransformerUnion`` way.

    Subclasses such as ``TransformerProduct`` reuse the union storage but define their own
    merge. Supporting them would require re-implementing that merge, so they are rejected
    explicitly rather than silently producing wrong features.
    """

    return type(obj).transform_one is union.TransformerUnion.transform_one


_CLASS_FLAGS_CACHE: dict[type, tuple[bool, bool, bool]] = {}


def _class_flags(obj) -> tuple[bool, bool, bool]:
    """Per-class structural flags, cached on the class.

    ``(is_anomaly_filter, is_active_learner, is_transformer_like)``. These only depend on
    the class' MRO and methods, so caching avoids re-walking the MRO for every node on
    every prediction.
    """

    klass = type(obj)
    cached = _CLASS_FLAGS_CACHE.get(klass)
    if cached is not None:
        return cached
    flags = (
        utils.inspect.ischildobject(obj=obj, class_name="AnomalyFilter"),
        utils.inspect.isactivelearner(obj),
        _is_transformer_like(obj),
    )
    _CLASS_FLAGS_CACHE[klass] = flags
    return flags


def _check_transformer_tree(obj, path: str, seen: set, leaves: list) -> None:
    """Validate a transformer subtree and collect its leaves.

    Raises before any step has a chance to learn or transform.
    """

    is_anomaly_filter, is_active_learner, transformer_like = _class_flags(obj)

    if is_anomaly_filter:
        raise ValueError(
            "compose.Pipeline prediction contexts are not supported with anomaly filters; "
            f"found one at {path}"
        )
    if is_active_learner:
        raise ValueError(
            "compose.Pipeline prediction contexts are not supported with active learners; "
            f"found one at {path}"
        )

    if isinstance(obj, Pipeline):
        if not obj.steps:
            raise ValueError(f"Empty nested pipeline at {path} cannot be used as a transformer")
        last = list(obj.steps.values())[-1]
        if not _is_transformer_like(last):
            raise ValueError(f"Nested pipeline at {path} ends with a predictor, not a transformer")
        if id(obj) in seen:
            raise ValueError(f"Pipeline at {path} contains an object graph cycle")
        seen.add(id(obj))
        for name, step in obj.steps.items():
            _check_transformer_tree(step, f"{path}.{name}", seen, leaves)
        seen.discard(id(obj))
        return

    if isinstance(obj, union.TransformerUnion):
        if not _has_builtin_union_semantics(obj):
            raise ValueError(
                f"Transformer union at {path} defines a custom merge and is not supported "
                "by the prediction-context protocol"
            )
        if id(obj) in seen:
            raise ValueError(f"TransformerUnion at {path} contains an object graph cycle")
        seen.add(id(obj))
        for name, sub_step in obj.transformers.items():
            _check_transformer_tree(sub_step, f"{path}.{name}", seen, leaves)
        seen.discard(id(obj))
        return

    if not transformer_like:
        raise ValueError(f"{path} is not a transformer; only supervised predictors may be terminal")
    leaves.append(obj)


def _check_shared_branches(leaves) -> None:
    """Reject a stateful leaf instance reached through more than one path.

    A stateful transformer shared by several branches would have to be recorded and
    updated once per branch for the same observation, which the protocol does not define.
    Stateless instances may be shared freely: applying them twice is idempotent.
    """

    counts: dict[int, int] = {}
    for leaf in leaves:
        if _is_stateful_step(leaf):
            counts[id(leaf)] = counts.get(id(leaf), 0) + 1
    if any(count > 1 for count in counts.values()):
        raise ValueError(
            "compose.Pipeline prediction contexts do not support sharing a stateful "
            "transformer instance between several branches"
        )


def _prediction_transform(context, transformer, x: dict) -> dict:
    """Run one transformer during the prediction phase.

    Unsupervised transformers learn from ``x`` and then transform it. Supervised
    transformers only transform; their input is snapshotted so that feedback can learn
    from exactly the same values. Nested pipelines and unions are traversed recursively,
    so every leaf runs exactly once.
    """

    if isinstance(transformer, Pipeline):
        current = x
        for step in transformer.steps.values():
            current = _prediction_transform(context, step, current)
        return current

    if isinstance(transformer, union.TransformerUnion):
        branch_outputs = [
            _prediction_transform(context, sub_step, x)
            for sub_step in transformer.transformers.values()
        ]
        return dict(collections.ChainMap(*branch_outputs))

    if _is_supervised_step(transformer):
        # Snapshot before transforming: the transformer may reuse or mutate a buffer, and
        # the caller may mutate its input dict right after this call.
        context._add_step(transformer, x)
        return transformer.transform_one(x)

    transformer.learn_one(x)
    return transformer.transform_one(x)


@contextlib.contextmanager
def learn_during_predict():
    """A context manager for fitting unsupervised steps during prediction.

    Usually, unsupervised parts of a pipeline are updated during `learn_one`. However, in the case
    of online learning, it is possible to update them before, during the prediction step. This
    context manager allows you to do so.

    This usually brings a slight performance improvement. But it is not done by default because it
    is not intuitive and is more difficult to test. It also means that you have to call
    `predict_one` before `learn_one` in order for the whole pipeline to be updated.

    Examples
    --------

    Let's first see what methods are called if we just call `predict_one`.

    >>> import io
    >>> import logging
    >>> from river import compose
    >>> from river import datasets
    >>> from river import linear_model
    >>> from river import preprocessing
    >>> from river import utils

    >>> model = compose.Pipeline(
    ...     preprocessing.StandardScaler(),
    ...     linear_model.LinearRegression()
    ... )

    >>> class_condition = lambda x: x.__class__.__name__ in ('StandardScaler', 'LinearRegression')

    >>> logger = logging.getLogger()
    >>> logger.setLevel(logging.DEBUG)

    >>> logs = io.StringIO()
    >>> sh = logging.StreamHandler(logs)
    >>> sh.setLevel(logging.DEBUG)
    >>> logger.addHandler(sh)

    >>> with utils.log_method_calls(class_condition):
    ...     for x, y in datasets.TrumpApproval().take(1):
    ...         _ = model.predict_one(x)

    >>> print(logs.getvalue())
    StandardScaler.transform_one
    LinearRegression.predict_one

    Now let's use the context manager and see what methods get called.

    >>> logs = io.StringIO()
    >>> sh = logging.StreamHandler(logs)
    >>> sh.setLevel(logging.DEBUG)
    >>> logger.addHandler(sh)

    >>> with utils.log_method_calls(class_condition), compose.learn_during_predict():
    ...     for x, y in datasets.TrumpApproval().take(1):
    ...         _ = model.predict_one(x)

    >>> print(logs.getvalue())
    StandardScaler.learn_one
    StandardScaler.transform_one
    LinearRegression.predict_one

    We can see that the scaler did not get updated before transforming the data.

    This also works when working with mini-batches.

    >>> logs = io.StringIO()
    >>> sh = logging.StreamHandler(logs)
    >>> sh.setLevel(logging.DEBUG)
    >>> logger.addHandler(sh)

    >>> with utils.log_method_calls(class_condition):
    ...     for x, y in datasets.TrumpApproval().take(1):
    ...         _ = model.predict_many(pd.DataFrame([x]))
    >>> print(logs.getvalue())
    StandardScaler.transform_many
    LinearRegression.predict_many

    >>> logs = io.StringIO()
    >>> sh = logging.StreamHandler(logs)
    >>> sh.setLevel(logging.DEBUG)
    >>> logger.addHandler(sh)

    >>> with utils.log_method_calls(class_condition), compose.learn_during_predict():
    ...     for x, y in datasets.TrumpApproval().take(1):
    ...         _ = model.predict_many(pd.DataFrame([x]))
    >>> print(logs.getvalue())
    StandardScaler.learn_many
    StandardScaler.transform_many
    LinearRegression.predict_many

    """
    Pipeline._LEARN_UNSUPERVISED_DURING_PREDICT = True
    try:
        yield
    finally:
        Pipeline._LEARN_UNSUPERVISED_DURING_PREDICT = False


class Pipeline(base.Estimator):
    """A pipeline of estimators.

    Pipelines allow you to chain different steps into a sequence. Typically, when doing supervised
    learning, a pipeline contains one or more transformation steps, whilst it's a regressor or
    a classifier. It is highly recommended to use pipelines with River. Indeed, in an online
    learning setting, it is very practical to have a model defined as a single object. Take a look
    at the [user guide](/recipes/pipelines) for further information and
    practical examples.

    One special thing to take notice to is the way transformers are handled. It is usual to predict
    something for a sample and wait for the ground truth to arrive. In such a scenario, the
    features are seen before the ground truth arrives. Therefore, the unsupervised parts of the
    pipeline are updated when `predict_one` and `predict_proba_one` are called. Usually the
    unsupervised parts of the pipeline are all the steps that precede the final step, which is a
    supervised model. However, some transformers are supervised and are therefore also updated
    during calls to `learn_one`.

    Parameters
    ----------
    steps
        Ideally, a list of (name, estimator) tuples. A name is automatically inferred if none is
        provided.

    Examples
    --------

    The recommended way to declare a pipeline is to use the `|` operator. The latter allows you
    to chain estimators in a very terse manner:

    >>> from river import linear_model
    >>> from river import preprocessing

    >>> scaler = preprocessing.StandardScaler()
    >>> log_reg = linear_model.LinearRegression()
    >>> model = scaler | log_reg

    This results in a pipeline that stores each step inside a dictionary.

    >>> model
    Pipeline (
      StandardScaler (
        with_std=True
      ),
      LinearRegression (
        optimizer=SGD (
          lr=Constant (
            learning_rate=0.01
          )
        )
        loss=Squared ()
        l2=0.
        l1=0.
        intercept_init=0.
        intercept_lr=Constant (
          learning_rate=0.01
        )
        clip_gradient=1e+12
        initializer=Zeros ()
      )
    )

    You can access parts of a pipeline in the same manner as a dictionary:

    >>> model['LinearRegression']
    LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
        )
      )
      loss=Squared ()
      l2=0.
      l1=0.
      intercept_init=0.
      intercept_lr=Constant (
        learning_rate=0.01
      )
      clip_gradient=1e+12
      initializer=Zeros ()
    )

    Note that you can also declare a pipeline by using the `compose.Pipeline` constructor
    method, which is slightly more verbose:

    >>> from river import compose

    >>> model = compose.Pipeline(scaler, log_reg)

    By using a `compose.TransformerUnion`, you can define complex pipelines that apply
    different steps to different parts of the data. For instance, we can extract word counts
    from text data, and extract polynomial features from numeric data.

    >>> from river import feature_extraction as fx

    >>> tfidf = fx.TFIDF('text')
    >>> counts = fx.BagOfWords('text')
    >>> text_part = compose.Select('text') | (tfidf + counts)

    >>> num_part = compose.Select('a', 'b') | fx.PolynomialExtender()

    >>> model = text_part + num_part
    >>> model |= preprocessing.StandardScaler()
    >>> model |= linear_model.LinearRegression()

    The following shows an example of using `debug_one` to visualize how the information
    flows and changes throughout the pipeline.

    >>> from river import compose
    >>> from river import naive_bayes

    >>> dataset = [
    ...     ('A positive comment', True),
    ...     ('A negative comment', False),
    ...     ('A happy comment', True),
    ...     ('A lovely comment', True),
    ...     ('A harsh comment', False)
    ... ]

    >>> tfidf = fx.TFIDF() | compose.Prefixer('tfidf_')
    >>> counts = fx.BagOfWords() | compose.Prefixer('count_')
    >>> mnb = naive_bayes.MultinomialNB()
    >>> model = (tfidf + counts) | mnb

    >>> for x, y in dataset:
    ...     model.learn_one(x, y)

    >>> x = dataset[0][0]
    >>> report = model.debug_one(dataset[0][0])
    >>> print(report)
    0. Input
    --------
    A positive comment
    1. Transformer union
    --------------------
        1.0 TFIDF | Prefixer
        --------------------
        tfidf_comment: 0.43017 (float)
        tfidf_positive: 0.90275 (float)
        1.1 BagOfWords | Prefixer
        -------------------------
        count_comment: 1 (int)
        count_positive: 1 (int)
    count_comment: 1 (int)
    count_positive: 1 (int)
    tfidf_comment: 0.43017 (float)
    tfidf_positive: 0.90275 (float)
    2. MultinomialNB
    ----------------
    False: 0.19221
    True: 0.80779

    """

    _LEARN_UNSUPERVISED_DURING_PREDICT = False

    # Set of ids of pipelines that currently have at least one pending prediction context.
    # This is used to decide whether a pipeline needs to carry an internal bookkeeping
    # table. See ``_pending_contexts`` below.
    _EMPTY_PENDING_IDS: typing.ClassVar[set] = set()

    def __init__(self, *steps):
        self.steps = collections.OrderedDict()
        for step in steps:
            self |= step

    def __getitem__(self, key):
        """Just for convenience."""
        if isinstance(key, int):
            return self.steps[list(self.steps.keys())[key]]
        return self.steps[key]

    def __len__(self):
        """Just for convenience."""
        return len(self.steps)

    def __or__(self, other):
        """Insert a step at the end of the pipeline."""
        self._add_step(other, at_start=False)
        return self

    def __ror__(self, other):
        """Insert a step at the start of the pipeline."""
        self._add_step(other, at_start=True)
        return self

    def __add__(self, other):
        """Merge with another Pipeline or TransformerUnion into a TransformerUnion."""
        if isinstance(other, union.TransformerUnion):
            return other.__add__(self)
        return union.TransformerUnion(self, other)

    def __mul__(self, other):
        from river import compose

        if isinstance(other, base.Transformer) or isinstance(other, Pipeline):
            return compose.TransformerProduct(self, other)

        return compose.Grouper(transformer=self, by=other)

    def __rmul__(self, other):
        return self * other

    def __str__(self):
        return " | ".join(map(str, self.steps.values()))

    def __repr__(self):
        return (
            "Pipeline (\n\t"
            + "\t".join(",\n".join(map(repr, self.steps.values())).splitlines(True))
            + "\n)"
        ).expandtabs(2)

    def _get_params(self):
        return {name: step._get_params() for name, step in self.steps.items()}

    def clone(self, new_params: dict | None = None, include_attributes=False):
        if new_params is None:
            new_params = {}

        return self.__class__(
            *[
                (name, new_params[name])
                if isinstance(new_params.get(name), base.Estimator)
                else (
                    name,
                    step.clone(new_params.get(name, {}), include_attributes=include_attributes),
                )
                for name, step in self.steps.items()
            ]
        )

    def mutate(self, new_params: dict):
        for step_name, step_params in new_params.items():
            self[step_name].mutate(step_params)

    @property
    def _supervised(self):
        return any(step._supervised for step in self.steps.values())

    @property
    def _last_step(self):
        return list(self.steps.values())[-1]

    @property
    def _multiclass(self):
        return self._last_step._multiclass

    def _add_step(self, obj: typing.Any, at_start: bool):
        """Add a step to either end of the pipeline.

        This method takes care of sanitizing the input. For instance, if a function is passed,
        then it will be wrapped with a `compose.FuncTransformer`.

        """

        name = None
        if isinstance(obj, tuple):
            name, obj = obj

        def _coerce_to_estimator(obj: typing.Any) -> base.Estimator:
            if isinstance(obj, types.FunctionType) or isinstance(obj, types.LambdaType):
                return func.FuncTransformer(obj)
            if isinstance(obj, list):
                return union.TransformerUnion(*[_coerce_to_estimator(part) for part in obj])
            return obj

        estimator = _coerce_to_estimator(obj)

        def infer_name(estimator: base.Estimator | typing.Callable) -> str:
            if isinstance(estimator, func.FuncTransformer):
                return infer_name(estimator.func)
            if isinstance(estimator, types.FunctionType) or isinstance(obj, types.LambdaType):
                return estimator.__name__  # type: ignore
            if hasattr(estimator, "__class__"):
                return estimator.__class__.__name__
            return str(estimator)

        # Infer a name if none is given
        if name is None:
            name = infer_name(estimator)

        # Make sure the name doesn't already exist
        if name in self.steps:
            counter = 1
            while f"{name}{counter}" in self.steps:
                counter += 1
            name = f"{name}{counter}"

        # Instantiate the estimator if it hasn't been done
        if isinstance(estimator, type):
            estimator = estimator()

        # Store the step
        self.steps[name] = estimator

        # Move the step to the start of the pipeline if so instructed
        if at_start:
            self.steps.move_to_end(name, last=False)

    # Single instance methods

    def learn_one(self, x: dict, y=None, **params):
        """Fit to a single instance.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A target value.

        """

        # Loop over the first n - 1 steps, which should all be transformers
        for step in self.steps.values():
            # There might be an anomaly filter in the pipeline. Its purpose is to prevent anomalous
            # data from being learned by the subsequent parts of the pipeline.
            if utils.inspect.ischildobject(obj=step, class_name="AnomalyFilter"):
                if step._supervised:
                    step.learn_one(x, y)
                    score = step.score_one(x, y)
                else:
                    step.learn_one(x)
                    score = step.score_one(x)
                # Skip the next parts of the pipeline if the score is classified as anomalous
                if step.classify(score):
                    break
                continue

            x_pre = x
            if isinstance(step, base.Transformer):
                # In case of _LEARN_UNSUPERVISED_DURING_PREDICT, then the unsupervised transformers
                # are updated before transforming.
                if not self._LEARN_UNSUPERVISED_DURING_PREDICT:
                    if isinstance(step, union.TransformerUnion):
                        for sub_step in step.transformers.values():
                            if not sub_step._supervised:
                                sub_step.learn_one(x)
                    elif not step._supervised:
                        step.learn_one(x)
                # Transform the data
                x = step.transform_one(x)

            # The supervised transformers have to be updated.
            # Note that this is done after transforming in order to avoid target leakage.
            if isinstance(step, base.Transformer):
                if isinstance(step, union.TransformerUnion):
                    for sub_step in step.transformers.values():
                        if sub_step._supervised:
                            sub_step.learn_one(x=x_pre, y=y)
                # Here the step is a supervised transformer, such as a TargetAgg. It's important
                # to pass the original features to the transformer, not the transformed ones.
                elif step._supervised:
                    step.learn_one(x=x_pre, y=y)  # type: ignore
            # Here the step is not a transformer, and it's supervised, such as a LinearRegression.
            # This is usually the last step of the pipeline.
            elif step._supervised:
                step.learn_one(x=x, y=y)
            # Here the step is not a transformer, and it's unsupervised, such as a KMeans. This
            # is also usually the last step of the pipeline.
            else:
                step.learn_one(x=x)

    def _transform_one(self, x: dict):
        """This methods takes care of applying the first n - 1 steps of the pipeline, which are
        supposedly transformers. It also returns the final step so that other functions can do
        something with it.

        """

        steps = iter(self.steps.values())

        for t in itertools.islice(steps, len(self) - 1):
            # An anomaly filter is a no-op during inference
            if utils.inspect.ischildobject(obj=t, class_name="AnomalyFilter"):
                continue

            # In case of _LEARN_UNSUPERVISED_DURING_PREDICT, then the unsupervised transformers
            # are updated during the inference phase.
            if self._LEARN_UNSUPERVISED_DURING_PREDICT:
                if isinstance(t, union.TransformerUnion):
                    for sub_t in t.transformers.values():
                        if not sub_t._supervised:
                            sub_t.learn_one(x)
                elif not t._supervised:
                    t.learn_one(x)

            x = t.transform_one(x)

        last_step = next(steps)
        return x, last_step

    def transform_one(self, x: dict, **params):
        """Apply each transformer in the pipeline to some features.

        The final step in the pipeline will be applied if it is a transformer. If not, then it will
        be ignored and the output from the penultimate step will be returned. Note that the steps
        that precede the final step are assumed to all be transformers.

        """
        x, last_step = self._transform_one(x)
        if isinstance(last_step, base.Transformer):
            if not last_step._supervised and self._LEARN_UNSUPERVISED_DURING_PREDICT:
                last_step.learn_one(x)
            return last_step.transform_one(x, **params)
        return x

    def predict_one(self, x: dict, **params):
        """Call `transform_one` on the first steps and `predict_one` on the last step.

        Parameters
        ----------
        x
            A dictionary of features.

        """
        x, last_step = self._transform_one(x)
        return last_step.predict_one(x, **params)

    def predict_proba_one(self, x: dict, **params):
        """Call `transform_one` on the first steps and `predict_proba_one` on the last step.

        Parameters
        ----------
        x
            A dictionary of features.

        """
        x, last_step = self._transform_one(x)
        return last_step.predict_proba_one(x, **params)

    # Prediction/feedback protocol ----------------------------------
    #
    # The methods below implement a two-phase view of a single observation:
    #
    # * Phase 1 (prediction) transforms the input. Every unsupervised transformer learns
    #   from the input first and then transforms it; supervised transformers only transform,
    #   so they cannot see ``y`` early; the terminal predictor only predicts.
    # * Phase 2 (feedback) learns from the exact feature values that were produced in
    #   phase 1. The terminal predictor sees the features it predicted with, and every
    #   supervised transformer sees the input that entered it during phase 1. Unsupervised
    #   steps are not learned again, and ``transform_one`` is never re-run.
    #
    # This is deliberately separate from ``learn_one``/``_transform_one`` so that the
    # historical arrival-time semantics are preserved bit for bit.

    def _check_context_support(self, predict_method: str = "predict_one"):
        """Reject pipelines that the two-phase protocol cannot support.

        This must run *before* any step learns or transforms, so that a rejected pipeline
        is left exactly as it was.
        """

        steps = list(self.steps.values())
        if not steps:
            raise ValueError("Cannot predict with an empty pipeline")

        leaves: list = []
        for step in steps[:-1]:
            _check_transformer_tree(step, "Pipeline", set(), leaves)

        terminal = steps[-1]
        if isinstance(terminal, Pipeline):
            raise ValueError(
                "compose.Pipeline prediction contexts require a predictor as the last "
                "step, not a nested pipeline"
            )
        if utils.inspect.ischildobject(obj=terminal, class_name="AnomalyFilter"):
            raise ValueError(
                "compose.Pipeline prediction contexts are not supported with anomaly "
                "filters; found one as the last step"
            )
        if utils.inspect.isactivelearner(terminal):
            raise ValueError(
                "compose.Pipeline prediction contexts are not supported with active "
                "learners; found one as the last step"
            )
        if _is_transformer_like(terminal):
            raise ValueError(
                "compose.Pipeline prediction contexts require the last step to be a "
                f"predictor, but {terminal!r} is a transformer"
            )
        if not _is_supervised_step(terminal):
            raise ValueError(
                "compose.Pipeline prediction contexts require a supervised last step, "
                f"but {terminal!r} is not supervised"
            )
        if predict_method == "predict_proba_one" and not callable(
            getattr(terminal, predict_method, None)
        ):
            raise ValueError(
                f"{terminal!r} does not implement {predict_method}; probability "
                "predictions are not available for this pipeline"
            )

        # A stateful leaf reached through two sibling branches would have to be recorded
        # and updated twice for the same observation. Reject that up-front. Sharing a
        # stateless instance is fine, because applying it twice is idempotent.
        _check_shared_branches(leaves)

    def _run_prediction(self, context, x, predict_method):
        """Run phase 1 over the pipeline tree, recording feedback snapshots."""

        steps = list(self.steps.values())
        current = x

        for t in steps[:-1]:
            current = _prediction_transform(context, t, current)

        return current, steps[-1]

    def _predict_with_context(self, x: dict, predict_method: str, **params):  # noqa: F811
        """Shared implementation of ``predict_one_with_context`` and its proba variant."""

        self._check_context_support(predict_method)

        context = _PredictionContext(owner=self)
        try:
            features, terminal = self._run_prediction(
                context=context, x=x, predict_method=predict_method
            )
        except BaseException:
            context._terminate(_PredictionContext.FAILED)
            raise

        # Snapshot before handing the features over: a predictor is allowed to mutate its
        # input, but the value used for feedback must be the one it was given to predict.
        snapshot = dict(features)

        try:
            prediction = getattr(terminal, predict_method)(features, **params)
        except BaseException:
            context._terminate(_PredictionContext.FAILED)
            raise

        if isinstance(prediction, dict):
            # Hand the caller a private copy: mutating the returned probabilities must not
            # be able to affect the prediction that is later scored.
            prediction = _clone_prediction(prediction)

        context._finish_predict(terminal=terminal, features=snapshot)
        self._register_context(context)
        return prediction, context

    def predict_one_with_context(self, x: dict, **params):
        """Predict and retain the features that were used to make the prediction.

        Returns a ``(prediction, context)`` pair. The context is an opaque, one-shot object
        that can be passed to :meth:`learn_one_from_context` once the target arrives, or
        released with :meth:`discard_context`.

        This is intended for regressors: it calls ``predict_one`` on the last step.

        The observation is split into two phases. During prediction, every unsupervised
        transformer learns from the input and then transforms it, supervised transformers
        only transform (so they cannot see ``y`` early), and the final predictor only
        predicts. During feedback, the final predictor learns from the exact features it
        predicted with, and each supervised transformer learns from the input that entered
        it during prediction.

        Examples
        --------

        >>> from river import compose, linear_model, preprocessing

        >>> model = compose.Pipeline(
        ...     preprocessing.StandardScaler(),
        ...     linear_model.LinearRegression()
        ... )

        >>> prediction, context = model.predict_one_with_context({'a': 1})
        >>> prediction
        0.0

        The scaler was updated while predicting:

        >>> dict(model['StandardScaler'].counts)
        {'a': 1}

        The label arrives later, and the pipeline is updated with the features it actually
        used to predict:

        >>> model.learn_one_from_context(context, 2.0)

        A context can only be learned from once:

        >>> model.discard_context(context)
        False

        """
        return self._predict_with_context(x, "predict_one", **params)

    def predict_proba_one_with_context(self, x: dict, **params):
        """Same as :meth:`predict_one_with_context`, but calls ``predict_proba_one``.

        This is intended for classifiers: the returned probabilities belong to the caller,
        and mutating them does not affect anything the pipeline keeps.

        Examples
        --------

        >>> from river import compose, linear_model, preprocessing

        >>> model = compose.Pipeline(
        ...     preprocessing.StandardScaler(),
        ...     linear_model.LogisticRegression()
        ... )

        >>> proba, context = model.predict_proba_one_with_context({'a': 1})
        >>> sorted(proba)
        [False, True]

        >>> model.learn_one_from_context(context, True)

        """
        return self._predict_with_context(x, "predict_proba_one", **params)

    def learn_one_from_context(self, context, y, **params):
        """Learn from the features captured by a prediction context.

        The terminal predictor is updated with the exact feature values it predicted with.
        Each supervised transformer is updated with the input that entered it during the
        prediction phase. Unsupervised transformers are not learned again, and
        ``transform_one`` is not re-run to rebuild the observation.

        Returns whatever the terminal predictor's ``learn_one`` returns, which is ``None``
        by convention in River.

        Raises
        ------
        ValueError
            If the context is not pending for this pipeline (forged, created by another
            pipeline, already learned from, or already discarded). No step is updated in
            that case. If a step raises during feedback, that original exception is
            propagated and the context is terminated; the steps that already ran are not
            replayed.

        """
        self._consume_context(context, y, params=params, learn=True)

    def discard_context(self, context) -> bool:
        """Release a pending context without learning anything.

        Returns ``True`` if the context was pending and has now been discarded, and
        ``False`` if it had already been learned from or discarded.

        Raises
        ------
        ValueError
            If the object was not created by this pipeline (or is not a context at all).
        """
        return self._consume_context(context, y=None, params=None, learn=False)

    # -- context bookkeeping --------------------------------------------

    def _register_context(self, context):
        pending = self.__dict__.get("_pending_contexts")
        if pending is None:
            pending = _PendingContexts(self, "_pending_contexts")
            self.__dict__["_pending_contexts"] = pending
        elif not pending._is_bound_to(self):
            # Inherited from a deepcopy: rebind to this pipeline and start empty.
            pending._bind(self, "_pending_contexts")
        pending.add(context)

    def _unregister_context(self, context):
        pending = self.__dict__.get("_pending_contexts")
        if pending is None:
            return
        pending.discard(context)
        if not pending:
            # No pending observation left: drop the bookkeeping structure entirely, so
            # that a pipeline does not retain a history of finished contexts.
            self.__dict__.pop("_pending_contexts", None)

    def _validate_context(self, context, allow_finished: bool = False):
        if not isinstance(context, _PredictionContext):
            raise ValueError(
                "Expected a prediction context produced by predict_*_with_context of this pipeline"
            )
        if not context._belongs_to(self):
            raise ValueError("This prediction context was not created by this pipeline instance")
        if not context._is_pending and not allow_finished:
            raise ValueError("This prediction context has already been consumed or discarded")

    def _consume_context(self, context, y, params, learn: bool):
        # Wrong-owner and forged objects always raise. A finished context is only an
        # error when it is used for learning; discarding it twice is a no-op returning
        # False.
        self._validate_context(context, allow_finished=not learn)

        if not context._is_pending:
            return False

        if not learn:
            context._terminate(_PredictionContext.DISCARDED)
            self._unregister_context(context)
            return True

        # Mark the context as being consumed *before* running any update. If a step
        # raises, the context is terminated in the ``except`` block below and can never
        # replay the steps that already ran.
        context._state = _PredictionContext.LEARNING
        steps, features, terminal = context._take_payload()
        # Release the pending bookkeeping entry up-front: whatever happens next, this
        # context will not be handed back to the caller for a second feedback.
        self._unregister_context(context)
        context._state = _PredictionContext.LEARNED

        try:
            for transformer, seen in steps:
                # ``y`` positional, as in ``SupervisedTransformer.learn_one``. Extra
                # feedback parameters are only forwarded to steps whose signature can
                # accept them (e.g. ``t`` for a rolling ``TargetAgg``).
                transformer.learn_one(seen, y, **_filter_learn_kwargs(transformer, params))
            terminal.learn_one(features, y, **params)
        except BaseException:
            # The partial update is *not* rolled back: third-party steps may have already
            # learned. The context is already terminal, so it cannot be replayed.
            context._state = _PredictionContext.FAILED
            raise
        return None

    def score_one(self, x: dict, **params):
        """Call `transform_one` on the first steps and `score_one` on the last step.

        Parameters
        ----------
        x
            A dictionary of features.

        """
        x, last_step = self._transform_one(x)
        return last_step.score_one(x, **params)

    def forecast(self, horizon: int, xs: list[dict] | None = None):
        """Return a forecast.

        Only works if each estimator has a `transform_one` method and the final estimator has a
        `forecast` method. This is the case of time series models from the `time_series` module.

        Parameters
        ----------
        horizon
            The forecast horizon.
        xs
            A list of features for each step in the horizon.

        """
        if xs is not None:
            xs = [self._transform_one(x)[0] for x in xs]
        return self._last_step.forecast(horizon=horizon, xs=xs)

    def debug_one(self, x: dict, show_types=True, n_decimals=5) -> str:
        """Displays the state of a set of features as it goes through the pipeline.

        Parameters
        ----------
        x
            A set of features.
        show_types
            Whether or not to display the type of feature along with it's value.
        n_decimals
            Number of decimals to display for each floating point value.

        """

        tab = " " * 4

        # We'll redirect all the print statement to a buffer, we'll return the content of the
        # buffer at the end
        buffer = io.StringIO()
        _print = functools.partial(print, file=buffer)

        def format_value(x):
            if isinstance(x, float):
                return "{:,.{prec}f}".format(x, prec=n_decimals)
            return x

        def print_dict(x, show_types, indent=False, space_after=True):
            # Some transformers accept strings as input instead of dicts
            if isinstance(x, str):
                _print(x)
            else:
                for k, v in sorted(x.items()):
                    type_str = f" ({type(v).__name__})" if show_types else ""
                    _print((tab if indent else "") + f"{k}: {format_value(v)}" + type_str)
            if space_after:
                _print()

        def print_title(title, indent=False):
            _print((tab if indent else "") + title)
            _print((tab if indent else "") + "-" * len(title))

        # Print the initial state of the features
        print_title("0. Input")
        print_dict(x, show_types=show_types)

        # Print the state of x at each step
        steps = iter(self.steps.values())
        for i, t in enumerate(itertools.islice(steps, len(self) - 1)):
            if isinstance(t, union.TransformerUnion):
                print_title(f"{i + 1}. Transformer union")
                for j, (name, sub_t) in enumerate(t.transformers.items()):
                    if isinstance(sub_t, Pipeline):
                        name = str(sub_t)
                    print_title(f"{i + 1}.{j} {name}", indent=True)
                    print_dict(sub_t.transform_one(x), show_types=show_types, indent=True)
                x = t.transform_one(x)
                print_dict(x, show_types=show_types)

            else:
                print_title(f"{i + 1}. {t}")
                x = t.transform_one(x)
                print_dict(x, show_types=show_types)

        # Print the predicted output from the final estimator
        final = next(steps)
        if not utils.inspect.istransformer(final):
            print_title(f"{len(self)}. {final}")

            # If the last estimator has a debug_one method then call it
            if hasattr(final, "debug_one"):
                _print(final.debug_one(x))

            # Display the prediction
            _print()
            if utils.inspect.isclassifier(final):
                print_dict(final.predict_proba_one(x), show_types=False, space_after=False)
            elif utils.inspect.isanomalydetector(final):
                _print(f"Score: {format_value(final.score_one(x))}")
            else:
                _print(f"Prediction: {format_value(final.predict_one(x))}")

        return buffer.getvalue().rstrip()

    # Mini-batch methods

    def learn_many(self, X: pd.DataFrame, y: pd.Series | None = None, **params):
        """Fit to a mini-batch.

        Parameters
        ----------
        X
            A dataframe of features. Columns can be added and/or removed between successive calls.
        y
            A series of target values.

        """

        steps = iter(self.steps.values())

        # Loop over the first n - 1 steps, which should all be transformers
        for t in itertools.islice(steps, len(self) - 1):
            if not self._LEARN_UNSUPERVISED_DURING_PREDICT:
                if isinstance(t, union.TransformerUnion):
                    for sub_t in t.transformers.values():
                        if not sub_t._supervised:
                            sub_t.learn_many(X)
                elif not t._supervised:
                    t.learn_many(X)

            X_pre = X
            X = t.transform_many(X=X)

            # The supervised transformers have to be updated.
            # Note that this is done after transforming in order to avoid target leakage.
            if isinstance(t, union.TransformerUnion):
                for sub_t in t.transformers.values():
                    if sub_t._supervised:
                        sub_t.learn_many(X=X_pre, y=y)

            elif t._supervised:
                t.learn_many(X=X_pre, y=y)

        last_step = next(steps)
        if last_step._supervised:
            last_step.learn_many(X=X, y=y, **params)
        else:
            last_step.learn_many(X=X, **params)

    def _transform_many(self, X: pd.DataFrame):
        """This methods takes care of applying the first n - 1 steps of the pipeline, which are
        supposedly transformers. It also returns the final step so that other functions can do
        something with it.

        """

        steps = iter(self.steps.values())

        for t in itertools.islice(steps, len(self) - 1):
            if self._LEARN_UNSUPERVISED_DURING_PREDICT:
                if isinstance(t, union.TransformerUnion):
                    for sub_t in t.transformers.values():
                        if not sub_t._supervised:
                            sub_t.learn_many(X=X)

                elif not t._supervised:
                    t.learn_many(X=X)

            X = t.transform_many(X=X)

        last_step = next(steps)
        return X, last_step

    def transform_many(self, X: pd.DataFrame):
        """Apply each transformer in the pipeline to some features.

        The final step in the pipeline will be applied if it is a transformer. If not, then it will
        be ignored and the output from the penultimate step will be returned. Note that the steps
        that precede the final step are assumed to all be transformers.

        """
        X, last_step = self._transform_many(X=X)
        if isinstance(last_step, base.MiniBatchTransformer):
            if not last_step._supervised and not self._LEARN_UNSUPERVISED_DURING_PREDICT:
                last_step.learn_many(X)
            return last_step.transform_many(X)
        return X

    def predict_many(self, X: pd.DataFrame):
        """Call transform_many, and then predict_many on the final step."""
        X, last_step = self._transform_many(X=X)
        return last_step.predict_many(X=X)

    def predict_proba_many(self, X: pd.DataFrame):
        """Call transform_many, and then predict_proba_many on the final step."""
        X, last_step = self._transform_many(X=X)
        return last_step.predict_proba_many(X=X)

    def _unit_test_skips(self):
        return self[-1]._unit_test_skips()
