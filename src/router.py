"""Dispatches a failure to the handler for its class.

The engine does not branch on failure class. It asks the router for a handler
and calls the same two functions on whatever comes back. That is what lets two
of the six classes be five lines each.

Dispatch is on the RETURN CODE, not on any label the generator attached, so
this is the same thing a production system would do with an acquirer response.
An unmapped or missing code routes to the ambiguous handler rather than
raising: real acquirers emit codes nobody documented, and "unknown" is itself a
class with a defined cautious response.

A consistency check runs at import: every class in ``rules.NEVER_RETRY_CLASSES``
must have a handler whose action set excludes RETRY. Keeping that invariant in
one place means adding a class cannot silently make a dead mandate retryable.
"""

from __future__ import annotations

from generator import rules
from generator.models import AgentFailureEvent, FailureEvent

from classes.afa_incomplete import AfaIncompleteHandler
from classes.base import FailureClassHandler
from classes.downtime import DowntimeHandler
from classes.expired_card import ExpiredCardHandler
from classes.insufficient_funds import InsufficientFundsHandler
from classes.opted_out import OptedOutHandler
from classes.stubs import AmbiguousHandler

_HANDLERS: dict[str, FailureClassHandler] = {
    rules.FAILURE_CLASS_INSUFFICIENT_FUNDS: InsufficientFundsHandler(),
    rules.FAILURE_CLASS_EXPIRED_CARD: ExpiredCardHandler(),
    rules.FAILURE_CLASS_OPTED_OUT: OptedOutHandler(),
    rules.FAILURE_CLASS_DOWNTIME: DowntimeHandler(),
    rules.FAILURE_CLASS_AFA_INCOMPLETE: AfaIncompleteHandler(),
    rules.FAILURE_CLASS_AMBIGUOUS: AmbiguousHandler(),
}


def handler_for_class(failure_class: str) -> FailureClassHandler:
    """The handler registered for a failure class."""
    try:
        return _HANDLERS[failure_class]
    except KeyError:
        raise ValueError(f"no handler registered for {failure_class!r}") from None


def route(failure_event: FailureEvent | AgentFailureEvent) -> FailureClassHandler:
    """The handler for this failure, chosen by its return code."""
    return handler_for_class(rules.classify(failure_event.return_code))


def route_return_code(return_code: str | None) -> FailureClassHandler:
    """The handler a bare return code dispatches to."""
    return handler_for_class(rules.classify(return_code))


def implemented_classes() -> set[str]:
    """Classes with a real model behind them. The rest are stubs."""
    return {name for name, h in _HANDLERS.items() if h.implemented}


def _check_registry_is_consistent() -> None:
    """Fail at import if the taxonomy and the handlers disagree.

    Three invariants, all cheap and all the kind of thing that rots quietly:
    every declared class has a handler; every handler answers to its own name;
    and nothing in NEVER_RETRY_CLASSES offers RETRY.
    """
    missing = rules.FAILURE_CLASSES - set(_HANDLERS)
    if missing:
        raise AssertionError(f"failure classes with no handler: {sorted(missing)}")

    for name, handler in _HANDLERS.items():
        if handler.failure_class != name:
            raise AssertionError(
                f"handler for {name!r} reports itself as {handler.failure_class!r}"
            )

    for name in rules.NEVER_RETRY_CLASSES:
        actions = _HANDLERS[name].allowed_actions(None)  # type: ignore[arg-type]
        if "RETRY" in actions:
            raise AssertionError(
                f"{name} is in NEVER_RETRY_CLASSES but its handler offers RETRY"
            )


_check_registry_is_consistent()
