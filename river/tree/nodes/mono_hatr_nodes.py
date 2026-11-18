"""Adaptive regression tree leaves with monotonic constraint support."""

from __future__ import annotations

from .hatr_nodes import AdaLeafRegMean
from .mono_htr_nodes import MonoLeafMean


class MonoAdaLeafRegMean(MonoLeafMean, AdaLeafRegMean):
    """Target-mean adaptive leaf used by the monotonic Hoeffding adaptive tree.

    Inherits the drift detector machinery of :class:`AdaLeafRegMean` and the monotonic
    clamped prediction of :class:`MonoLeafMean`.
    """

    def __init__(self, stats, depth, splitter, drift_detector, rng, **kwargs):
        super().__init__(stats, depth, splitter, drift_detector=drift_detector, rng=rng, **kwargs)
