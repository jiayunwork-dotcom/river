"""One-shot prediction contexts.

This module defines the internal object that carries the feature values a pipeline saw
during a prediction, so that a supervised update can later be applied without re-running
``transform_one`` and without re-computing the prediction.

The object is deliberately opaque: the public API of ``compose.Pipeline`` is the only
supported way to create, consume, or discard it.
"""

from __future__ import annotations

import sys
import weakref

__all__ = ["_PredictionContext"]


class _PredictionContext:
    """Feature snapshots captured while a pipeline makes a prediction.

    A context has exactly one useful transition: it starts *pending*, and it either gets
    consumed by ``learn_one_from_context`` or released by ``discard_context``. Both
    transitions are terminal. Terminal contexts keep their owner reference so that a
    context handed to the wrong pipeline can still be told apart from a finished one, but
    they release every feature payload they were holding.
    """

    __slots__ = (
        "_owner_ref",
        "_state",
        "_steps",
        "_features",
        "_terminal",
        "__weakref__",
    )

    # State machine values. Anything other than PENDING is finished.
    PENDING = "pending"
    LEARNING = "learning"
    LEARNED = "learned"
    DISCARDED = "discarded"
    FAILED = "failed"

    def __init__(self, owner):
        # A context never keeps its pipeline alive. This also means that a context whose
        # owner has been garbage-collected can be recognised as belonging to a dead
        # pipeline, which is just another form of a wrong-owner context.
        self._owner_ref = weakref.ref(owner)
        self._state = self.PENDING
        # Pairs of (supervised transformer, input seen during prediction), in the order in
        # which the transformer was executed.
        self._steps: list = []
        # The exact feature values that were handed to the terminal predictor.
        self._features = None
        # The terminal predictor itself. Keeping the object (instead of re-deriving it
        # from the pipeline) means that feedback updates the object that actually made the
        # prediction, even if the pipeline is later mutated.
        self._terminal = None

    @property
    def _owner(self):
        return self._owner_ref()

    def _belongs_to(self, owner) -> bool:
        return self._owner_ref() is owner

    # -- capture ---------------------------------------------------------

    def _add_step(self, transformer, x: dict) -> None:
        # The snapshot has to happen *before* transform_one is called, because a
        # transformer is allowed to reuse/mutate buffers.
        self._steps.append((transformer, dict(x)))

    def _finish_predict(self, terminal, features: dict) -> None:
        self._features = dict(features)
        self._terminal = terminal

    # -- consumption -----------------------------------------------------

    def _take_payload(self):
        """Detach the pending payload. The caller becomes its sole owner."""
        steps = self._steps
        features = self._features
        terminal = self._terminal
        self._steps = []
        self._features = None
        self._terminal = None
        return steps, features, terminal

    def _terminate(self, state: str) -> None:
        self._state = state
        self._steps = []
        self._features = None
        self._terminal = None

    # -- introspection (tests/diagnostics) -------------------------------

    @property
    def _is_pending(self) -> bool:
        return self._state == self.PENDING

    def _payload_size(self) -> int:
        """Approximate size of the retained feature payload, in bytes.

        This is intentionally conservative and shallow: it accounts for the dictionaries
        and their immediate keys/values, which is what the context copies. Deep object
        graphs are not walked, because doing so per observation would defeat the point of
        the measurement.
        """

        total = 0
        for _, x in self._steps:
            total += _dict_size(x)
        if self._features is not None:
            total += _dict_size(self._features)
        return total

    def __repr__(self):
        return f"<PredictionContext state={self._state!r} owner={id(self._owner):#x}>"


def _dict_size(x: dict) -> int:
    size = sys.getsizeof(x)
    for k, v in x.items():
        size += sys.getsizeof(k)
        size += sys.getsizeof(v)
    return size
