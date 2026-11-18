# Unreleased

## compose

- Add `compose.Pipeline.predict_one_with_context` and
  `compose.Pipeline.predict_proba_one_with_context`. Both return a `(prediction, context)`
  pair. The context records the feature values that were actually used to make the
  prediction, so that `compose.Pipeline.learn_one_from_context(context, y)` can update the
  pipeline with exactly those values when the target arrives. `discard_context(context)`
  releases a pending context without learning; it returns `True` the first time and
  `False` afterwards. A context is one-shot and belongs to the pipeline that created it;
  forged, foreign, already-learned and already-discarded contexts raise `ValueError`
  without updating any step.
- Add `evaluate.iter_progressive_val_score(..., feature_view=...)` and
  `evaluate.progressive_val_score(..., feature_view=...)`. The default `'arrival'` keeps
  the existing behaviour. The new `'prediction'` value uses the context protocol so that
  each update is fed the features that were used for the prediction, while metrics are
  still updated only at answer events. Unsupported models and invalid values are rejected
  before the dataset is consumed.
- Contexts are only supported for trees made of ordinary transformers,
  `base.SupervisedTransformer`, nested transformation pipelines and
  `compose.TransformerUnion`, ending in a supervised regressor or classifier. Pipelines
  containing anomaly filters or active learners, and pipelines that share a *stateful*
  transformer instance between branches, are rejected before any step learns or
  transforms. The historical `learn_one` / `predict_one` interface is unchanged.

## evaluate

- `progressive_val_score` and `iter_progressive_val_score` gained the `feature_view`
  parameter described above. Both now release any pending contexts and close the input
  iterator when they finish, fail, or are closed explicitly.

## tree

- `HoeffdingTreeRegressor` and `HoeffdingAdaptiveTreeRegressor` accept a
  `monotonic_cst` mapping enforcing non-decreasing (`1`) or non-increasing (`-1`)
  predictions along selected features, for example `monotonic_cst={"income": 1,
  "debt": -1}`. Constraints require `leaf_prediction="mean"` and `binary_split=True`;
  other combinations, non-integer directions, booleans, and nominal-attribute
  conflicts raise `ValueError` before training. Constrained leaves keep learning their
  running means, while an isotonic coordinator projects each prediction onto a feasible
  interval derived from a graph of region-ordering and adaptive-tree tie constraints.
  The adaptive tree's drift detection, alternate-subtree training, switches and pruning
  remain active, and the partial order is preserved across every subtree replacement.
  Neutral configurations (`None`, an empty mapping, or all-zero directions) keep the
  original unconstrained behavior and structure.
