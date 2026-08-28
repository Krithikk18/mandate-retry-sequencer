"""The interface every failure class implements.

CONTEXT sets the contract: each class exposes exactly two things, and the
decision engine calls them without knowing which class it is handling.

    success_probability(state, day) -> float
    allowed_actions(state)          -> list of actions

The signature takes ``PolicyState`` rather than a bare customer because the
insufficient-funds model infers a payday from attempt history, and history is
the only evidence a PSP has. Everything in ``PolicyState`` is agent-visible.

The point of the abstraction is that two of the six classes need almost no
code. An expired card and a revoked mandate cannot succeed on any date, so
their success model is the constant zero and their action set simply omits
RETRY. A fixed schedule retries them four times anyway.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from policy.base import Action, PolicyState


class FailureClassHandler(ABC):
    """A success model and an action set for one class of failure."""

    #: canonical name from rules.FAILURE_CLASSES
    failure_class: str = ""
    #: False for classes that are stubs rather than models
    implemented: bool = True
    #: one line a human can read in an audit trail
    rationale: str = ""

    @abstractmethod
    def success_probability(self, state: PolicyState, day: date) -> float:
        """P(a presentment on ``day`` clears), given what the agent can see."""

    @abstractmethod
    def allowed_actions(self, state: PolicyState) -> list[Action]:
        """Actions the engine may choose from for this class."""

    def permits_retry(self, state: PolicyState | None = None) -> bool:
        """Whether RETRY is available at all for this class."""
        return "RETRY" in self.allowed_actions(state)  # type: ignore[arg-type]

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"<{type(self).__name__} {self.failure_class}>"


class NeverRetryHandler(FailureClassHandler):
    """Base for classes where no date can ever succeed.

    The success model is not "low", it is exactly zero, and that is a
    structural fact about the mandate rather than a statement about the
    customer's bank balance. Retrying cannot help, so RETRY is not offered.
    """

    def success_probability(self, state: PolicyState, day: date) -> float:
        return 0.0
