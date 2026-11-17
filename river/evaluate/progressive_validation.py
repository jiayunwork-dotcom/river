from __future__ import annotations

import datetime as dt
import itertools
import time
import typing

from river import base, metrics, stream, utils

__all__ = ["progressive_val_score"]

_FEATURE_VIEWS = ("arrival", "prediction")


def _check_feature_view(feature_view: str) -> None:
    if feature_view not in _FEATURE_VIEWS:
        raise ValueError(f"feature_view must be one of {_FEATURE_VIEWS}, got {feature_view!r}")


def _supports_prediction_view(model, pred_func_name: str) -> bool:
    required = ("learn_one_from_context", "discard_context", f"{pred_func_name}_with_context")
    return all(callable(getattr(model, method, None)) for method in required)


def _feature_view_prediction(model, x, pred_func_name, params):
    """Make a prediction with a context, returning (prediction, context)."""
    method = getattr(model, f"{pred_func_name}_with_context")
    return method(x, **params)


def _progressive_validation(
    dataset: base.typing.Dataset,
    model,
    metric: metrics.base.Metric,
    checkpoints: typing.Iterator[int],
    moment: str | typing.Callable[[dict], dt.datetime] | None = None,
    delay: str | int | dt.timedelta | typing.Callable | None = None,
    measure_time=False,
    measure_memory=False,
    yield_predictions=False,
    feature_view: str = "arrival",
):
    _check_feature_view(feature_view)

    # Check that the model and the metric are in accordance
    if not metric.works_with(model):
        raise ValueError(f"{metric.__class__.__name__} metric is not compatible with {model}")

    # Determine if predict_one or predict_proba_one should be used in case of a classifier
    if utils.inspect.isanomalydetector(model) or utils.inspect.isanomalyfilter(model):
        pred_func_name = "score_one"
    elif utils.inspect.isclassifier(model) and not metric.requires_labels:  # type: ignore
        pred_func_name = "predict_proba_one"
    else:
        pred_func_name = "predict_one"

    if feature_view == "prediction":
        if not _supports_prediction_view(model, pred_func_name):
            raise ValueError(
                "feature_view='prediction' requires a compose.Pipeline supporting the "
                "prediction-context protocol (predict_*_with_context / "
                "learn_one_from_context / discard_context); "
                f"{model!r} does not"
            )
        if (
            utils.inspect.isanomalydetector(model)
            or utils.inspect.isanomalyfilter(model)
            or utils.inspect.isactivelearner(model)
        ):
            raise ValueError(
                "feature_view='prediction' does not support anomaly filters, anomaly "
                f"detectors, or active learners; {model!r} uses one of them"
            )
        # Let the pipeline perform its own structural validation now, before the dataset
        # is touched (nested anomaly filters, shared stateful instances, etc.).
        validate = getattr(model, "_check_context_support", None)
        if callable(validate):
            validate(pred_func_name)

    preds = {}

    # If we are dealing with an active learner, we need to check whether or not a label should be
    # used for training or not. We'll also record how many times labels were used.
    active_learning = utils.inspect.isactivelearner(model)
    n_samples_learned = 0

    prev_checkpoint = None
    next_checkpoint = next(checkpoints, None)
    n_total_answers = 0
    if measure_time:
        start = time.perf_counter()

    def report(y_pred):
        if isinstance(metric, metrics.base.Metrics):
            state = {m.__class__.__name__: m for m in metric}
        else:
            state = {metric.__class__.__name__: metric}
        state["Step"] = n_total_answers
        if active_learning:
            state["Samples used"] = n_samples_learned
        if measure_time:
            now = time.perf_counter()
            state["Time"] = dt.timedelta(seconds=now - start)
        if measure_memory:
            state["Memory"] = model._raw_memory_usage
        if yield_predictions:
            state["Prediction"] = y_pred

        return state

    def run_events():
        nonlocal n_total_answers, prev_checkpoint, next_checkpoint, n_samples_learned

        events = stream.simulate_qa(dataset, moment, delay, copy=True)
        try:
            for i, x, y, *kwargs in events:
                kwargs = kwargs[0] if kwargs else {}

                # Case 1: no ground truth, just make a prediction
                if y is None:
                    if feature_view == "prediction":
                        y_pred, context = _feature_view_prediction(model, x, pred_func_name, kwargs)
                        preds[i] = (y_pred, True, context)
                    else:
                        y_pred = getattr(model, pred_func_name)(x, **kwargs)
                        y_pred, ask_for_label = y_pred if active_learning else (y_pred, True)
                        if utils.inspect.isanomalyfilter(model):
                            y_pred = model.classify(y_pred)
                        preds[i] = (y_pred, ask_for_label, None)
                    continue

                # Case 2: there's a ground truth, model and metric can be updated
                y_pred, use_label, context = preds.pop(i)

                # Update the metric. This happens at the answer event, using the
                # prediction that was actually issued.
                if y_pred != {} and y_pred is not None:
                    metric.update(y_true=y, y_pred=y_pred)

                # Update the model
                if use_label:
                    n_samples_learned += 1
                    if feature_view == "prediction":
                        # The failure already terminated the context; discarding is a no-op
                        # and keeps the evaluator's own bookkeeping in sync.
                        try:
                            model.learn_one_from_context(context, y, **kwargs)
                        finally:
                            model.discard_context(context)
                    elif model._supervised:
                        model.learn_one(x, y, **kwargs)
                    else:
                        model.learn_one(x, **kwargs)

                # Yield current results
                n_total_answers += 1
                if n_total_answers == next_checkpoint:
                    yield report(y_pred=y_pred)
                    prev_checkpoint = next_checkpoint
                    next_checkpoint = next(checkpoints, None)
            else:
                # If the dataset was exhausted, we need to make sure that we yield the final
                # results
                if prev_checkpoint and n_total_answers != prev_checkpoint:
                    yield report(y_pred=None)
        finally:
            # Release every context this evaluator still owns, as well as the input
            # iterator. This runs on normal exhaustion, on error, and on an explicit
            # ``close()`` of the generator.
            try:
                if feature_view == "prediction":
                    for _, _, context in list(preds.values()):
                        try:
                            model.discard_context(context)
                        except ValueError:
                            # Already terminated elsewhere; nothing left to release.
                            pass
                preds.clear()
            finally:
                close = getattr(events, "close", None)
                if callable(close):
                    close()
                close = getattr(dataset, "close", None)
                if callable(close):
                    close()

    yield from run_events()


def iter_progressive_val_score(
    dataset: base.typing.Dataset,
    model,
    metric: metrics.base.Metric,
    moment: str | typing.Callable | None = None,
    delay: str | int | dt.timedelta | typing.Callable | None = None,
    step=1,
    measure_time=False,
    measure_memory=False,
    yield_predictions=False,
    feature_view: str = "arrival",
) -> typing.Generator:
    """Evaluates the performance of a model on a streaming dataset and yields results.

    This does exactly the same as `evaluate.progressive_val_score`. The only difference is that
    this function returns an iterator, yielding results at every step. This can be useful if you
    want to have control over what you do with the results. For instance, you might want to plot
    the results.

    Parameters
    ----------
    dataset
        The stream of observations against which the model will be evaluated.
    model
        The model to evaluate.
    metric
        The metric used to evaluate the model's predictions.
    moment
        The attribute used for measuring time. If a callable is passed, then it is expected to take
        as input a `dict` of features. If `None`, then the observations are implicitly timestamped
        in the order in which they arrive.
    delay
        The amount to wait before revealing the target associated with each observation to the
        model. This value is expected to be able to sum with the `moment` value. For instance, if
        `moment` is a `datetime.date`, then `delay` is expected to be a `datetime.timedelta`. If a
        callable is passed, then it is expected to take as input a `dict` of features and the
        target. If a `str` is passed, then it will be used to access the relevant field from the
        features. If `None` is passed, then no delay will be used, which leads to doing standard
        online validation.
    step
        Iteration number at which to yield results. This only takes into account the
        predictions, and not the training steps.
    measure_time
        Whether or not to measure the elapsed time.
    measure_memory
        Whether or not to measure the memory usage of the model.
    yield_predictions
        Whether or not to include predictions. If step is 1, then this is equivalent to yielding
        the predictions at every iterations. Otherwise, not all predictions will be yielded.
    feature_view
        How features are presented to the model when it learns.

        - ``'arrival'`` (default) transforms the features when the observation arrives and
          learns with the arrival-time features, exactly as before.
        - ``'prediction'`` keeps the feature values that were actually used to make each
          prediction, and feeds them back when the label arrives. This requires a
          ``compose.Pipeline`` that supports the prediction-context protocol, and is
          rejected (before the dataset is consumed) otherwise.

        In both modes the metric is updated only at answer events, with the prediction that
        was actually issued.

    Examples
    --------

    Take the following model:

    >>> from river import linear_model
    >>> from river import preprocessing

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression()
    ... )

    We can evaluate it on the `Phishing` dataset as so:

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics

    >>> steps = evaluate.iter_progressive_val_score(
    ...     model=model,
    ...     dataset=datasets.Phishing(),
    ...     metric=metrics.ROCAUC(),
    ...     step=200
    ... )

    >>> for step in steps:
    ...     print(step)
    {'ROCAUC': ROCAUC: 90.20%, 'Step': 200}
    {'ROCAUC': ROCAUC: 92.25%, 'Step': 400}
    {'ROCAUC': ROCAUC: 93.23%, 'Step': 600}
    {'ROCAUC': ROCAUC: 94.05%, 'Step': 800}
    {'ROCAUC': ROCAUC: 94.79%, 'Step': 1000}
    {'ROCAUC': ROCAUC: 95.07%, 'Step': 1200}
    {'ROCAUC': ROCAUC: 95.07%, 'Step': 1250}

    The `yield_predictions` parameter can be used to include the predictions in the results:

    >>> import itertools

    >>> steps = evaluate.iter_progressive_val_score(
    ...     model=model,
    ...     dataset=datasets.Phishing(),
    ...     metric=metrics.ROCAUC(),
    ...     step=1,
    ...     yield_predictions=True
    ... )

    >>> for step in itertools.islice(steps, 100, 105):
    ...    print(step)
    {'ROCAUC': ROCAUC: 94.68%, 'Step': 101, 'Prediction': {False: 0.966..., True: 0.033...}}
    {'ROCAUC': ROCAUC: 94.75%, 'Step': 102, 'Prediction': {False: 0.035..., True: 0.964...}}
    {'ROCAUC': ROCAUC: 94.82%, 'Step': 103, 'Prediction': {False: 0.043..., True: 0.956...}}
    {'ROCAUC': ROCAUC: 94.89%, 'Step': 104, 'Prediction': {False: 0.816..., True: 0.183...}}
    {'ROCAUC': ROCAUC: 94.96%, 'Step': 105, 'Prediction': {False: 0.041..., True: 0.958...}}

    References
    ----------
    [^1]: [Beating the Hold-Out: Bounds for K-fold and Progressive Cross-Validation](http://hunch.net/~jl/projects/prediction_bounds/progressive_validation/coltfinal.pdf)
    [^2]: [Grzenda, M., Gomes, H.M. and Bifet, A., 2019. Delayed labelling evaluation for data streams. Data Mining and Knowledge Discovery, pp.1-30](https://link.springer.com/content/pdf/10.1007%2Fs10618-019-00654-y.pdf)

    """

    yield from _progressive_validation(
        dataset,
        model,
        metric,
        checkpoints=itertools.count(step, step) if step else iter([]),
        moment=moment,
        delay=delay,
        measure_time=measure_time,
        measure_memory=measure_memory,
        yield_predictions=yield_predictions,
        feature_view=feature_view,
    )


def progressive_val_score(
    dataset: base.typing.Dataset,
    model,
    metric: metrics.base.Metric,
    moment: str | typing.Callable | None = None,
    delay: str | int | dt.timedelta | typing.Callable | None = None,
    print_every=0,
    show_time=False,
    show_memory=False,
    feature_view: str = "arrival",
    **print_kwargs,
) -> metrics.base.Metric:
    """Evaluates the performance of a model on a streaming dataset.

    This method is the canonical way to evaluate a model's performance. When used correctly, it
    allows you to exactly assess how a model would have performed in a production scenario.

    `dataset` is converted into a stream of questions and answers. At each step the model is either
    asked to predict an observation, or is either updated. The target is only revealed to the model
    after a certain amount of time, which is determined by the `delay` parameter. Note that under
    the hood this uses the `stream.simulate_qa` function to go through the data in arrival order.

    By default, there is no delay, which means that the samples are processed one after the other.
    When there is no delay, this function essentially performs progressive validation. When there
    is a delay, then we refer to it as delayed progressive validation.

    It is recommended to use this method when you want to determine a model's performance on a
    dataset. In particular, it is advised to use the `delay` parameter in order to get a reliable
    assessment. Indeed, in a production scenario, it is often the case that ground truths are made
    available after a certain amount of time. By using this method, you can reproduce this scenario
    and therefore truthfully assess what would have been the performance of a model on a given
    dataset.

    Parameters
    ----------
    dataset
        The stream of observations against which the model will be evaluated.
    model
        The model to evaluate.
    metric
        The metric used to evaluate the model's predictions.
    moment
        The attribute used for measuring time. If a callable is passed, then it is expected to take
        as input a `dict` of features. If `None`, then the observations are implicitly timestamped
        in the order in which they arrive.
    delay
        The amount to wait before revealing the target associated with each observation to the
        model. This value is expected to be able to sum with the `moment` value. For instance, if
        `moment` is a `datetime.date`, then `delay` is expected to be a `datetime.timedelta`. If a
        callable is passed, then it is expected to take as input a `dict` of features and the
        target. If a `str` is passed, then it will be used to access the relevant field from the
        features. If `None` is passed, then no delay will be used, which leads to doing standard
        online validation.
    print_every
        Iteration number at which to print the current metric. This only takes into account the
        predictions, and not the training steps.
    show_time
        Whether or not to display the elapsed time.
    show_memory
        Whether or not to display the memory usage of the model.
    feature_view
        How features are presented to the model when it learns. ``'arrival'`` (the default)
        preserves the historical behaviour: features are transformed when the observation
        arrives, and the model learns with those arrival-time values. ``'prediction'`` uses
        `compose.Pipeline`'s two-phase context protocol so that each model update is fed
        the exact features that were used for the prediction, which matters when
        transformers evolve between the question and the answer. The latter requires a
        supported pipeline and is rejected before any data is consumed otherwise.
    print_kwargs
        Extra keyword arguments are passed to the `print` function. For instance, this allows
        providing a `file` argument, which indicates where to output progress.

    Examples
    --------

    Take the following model:

    >>> from river import linear_model
    >>> from river import preprocessing

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression()
    ... )

    We can evaluate it on the `Phishing` dataset as so:

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics

    >>> evaluate.progressive_val_score(
    ...     model=model,
    ...     dataset=datasets.Phishing(),
    ...     metric=metrics.ROCAUC(),
    ...     print_every=200
    ... )
    [200] ROCAUC: 90.20%
    [400] ROCAUC: 92.25%
    [600] ROCAUC: 93.23%
    [800] ROCAUC: 94.05%
    [1,000] ROCAUC: 94.79%
    [1,200] ROCAUC: 95.07%
    [1,250] ROCAUC: 95.07%
    ROCAUC: 95.07%

    We haven't specified a delay, therefore this is strictly equivalent to the following piece
    of code:

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression()
    ... )

    >>> metric = metrics.ROCAUC()

    >>> for x, y in datasets.Phishing():
    ...     y_pred = model.predict_proba_one(x)
    ...     metric.update(y, y_pred)
    ...     model.learn_one(x, y)

    >>> metric
    ROCAUC: 95.07%

    When `print_every` is specified, the current state is printed at regular intervals. Under
    the hood, Python's `print` method is being used. You can pass extra keyword arguments to
    modify its behavior. For instance, you may use the `file` argument if you want to log the
    progress to a file of your choice.

    >>> with open('progress.log', 'w') as f:
    ...     metric = evaluate.progressive_val_score(
    ...         model=model,
    ...         dataset=datasets.Phishing(),
    ...         metric=metrics.ROCAUC(),
    ...         print_every=200,
    ...         file=f
    ...     )

    >>> with open('progress.log') as f:
    ...     for line in f.read().splitlines():
    ...         print(line)
    [200] ROCAUC: 94.00%
    [400] ROCAUC: 94.70%
    [600] ROCAUC: 95.17%
    [800] ROCAUC: 95.42%
    [1,000] ROCAUC: 95.82%
    [1,200] ROCAUC: 96.00%
    [1,250] ROCAUC: 96.04%

    Note that the performance is slightly better than above because we haven't used a fresh
    copy of the model. Instead, we've reused the existing model which has already done a full
    pass on the data.

    >>> import os; os.remove('progress.log')

    The ``feature_view`` parameter controls which features the model learns with. By
    default (``'arrival'``) features are transformed when the observation arrives. With
    ``'prediction'``, each update uses the features that were actually used for the
    prediction, even if transformers changed while the label was in flight:

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LinearRegression()
    ... )
    >>> metric = evaluate.progressive_val_score(
    ...     model=model,
    ...     dataset=[({'x': i}, i) for i in range(20)],
    ...     metric=metrics.MAE(),
    ...     feature_view='prediction',
    ... )
    >>> metric.get() >= 0
    True

    References
    ----------
    [^1]: [Beating the Hold-Out: Bounds for K-fold and Progressive Cross-Validation](http://hunch.net/~jl/projects/prediction_bounds/progressive_validation/coltfinal.pdf)
    [^2]: [Grzenda, M., Gomes, H.M. and Bifet, A., 2019. Delayed labelling evaluation for data streams. Data Mining and Knowledge Discovery, pp.1-30](https://link.springer.com/content/pdf/10.1007%2Fs10618-019-00654-y.pdf)

    """

    checkpoints = iter_progressive_val_score(
        dataset=dataset,
        model=model,
        metric=metric,
        moment=moment,
        delay=delay,
        step=print_every,
        measure_time=show_time,
        measure_memory=show_memory,
        feature_view=feature_view,
    )

    active_learning = utils.inspect.isactivelearner(model)

    try:
        for checkpoint in checkpoints:
            msg = f"[{checkpoint['Step']:,d}] {metric}"
            if active_learning:
                msg += f" – {checkpoint['Samples used']:,d} samples used"
            if show_time:
                H, rem = divmod(checkpoint["Time"].seconds, 3600)
                M, S = divmod(rem, 60)
                msg += f" – {H:02d}:{M:02d}:{S:02d}"
            if show_memory:
                msg += f" – {utils.pretty.humanize_bytes(checkpoint['Memory'])}"
            print(msg, **print_kwargs)
    finally:
        # Closing the checkpoint generator releases the evaluator's pending contexts and
        # input iterator, both on normal completion and on an exception.
        close = getattr(checkpoints, "close", None)
        if callable(close):
            close()

    return metric
