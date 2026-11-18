"""Regression tree leaves with monotonic constraint support."""

from __future__ import annotations

from .htr_nodes import LeafMean


class MonoLeafMean(LeafMean):
    """Target-mean leaf used by monotonic Hoeffding tree regressors.

    The leaf behaves exactly like :class:`LeafMean` while learning; at prediction time its
    raw target mean is projected onto the feasible interval maintained by the tree's
    monotonic coordinator, so the global prediction function honors every constrained
    direction even when fresh labels push this leaf against the trend.
    """

    def learn_one(self, x, y, *, w=1.0, tree=None, **kwargs):
        super().learn_one(x, y, w=w, tree=tree, **kwargs)
        if tree is not None and getattr(tree, "_mono_active", False):
            tree._mono_touch(self)

    def prediction(self, x, *, tree=None):
        raw = self.stats.mean.get()
        return tree._mono_coordinator.clamp(self, raw)
