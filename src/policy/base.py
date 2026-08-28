"""The contract every retry policy implements.

A policy sees ``PolicyState`` and returns ``RetryDecision``. That is the whole
interface: the baseline, the oracle ceiling and the intelligent agent are
interchangeable behind it, so the harness can compare them without knowing
which is which.

``PolicyState`` is built exclusively from ``agent_view`` output. It holds
``Agent*`` records, which have no fields for the hidden generative parameters,
so a policy cannot reach the truth even by accident - and pydantic refuses to
coerce a truth-family ``Customer`` into an ``AgentCustomer`` slot, so it cannot
be smuggled in either. tests assert both.

The 24-hour commitment lag is the reason this interface looks the way it does.
Because RBI requires the issuer to notify at least a day before every debit, a
policy must NAME A DATE it cannot yet see the balance for. It is a scheduler
under a lead-time constraint, not a controller that reacts to a salary credit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from generator import rules
from generator.models import (
    AgentCustomer,
    AgentFailureEvent,
    AgentPaymentHistoryRecord,
)

Action = Literal["RETRY", "NUDGE", "WAIT", "HOLD", "CANCEL"]

TERMINAL_ACTIONS: frozenset[str] = frozenset({"HOLD", "CANCEL"})

# A presentment is a date plus a time, but the execution window is a rule about
# the time. Policies name the date; this is the hour the presentment is filed
# unless a policy deliberately overrides it. Inside the NPCI 00:00-07:00 window.
DEFAULT_PRESENTMENT_TIME: time = time(2, 0)


class RetryDecision(BaseModel):
    """What a policy decided to do today, and the numbers behind it.

    ``reasoning`` is the audit trail. Razorpay's brief asks for one explicitly,
    and it is also what makes a disagreement between two policies legible: the
    log says which believed what, not merely which won.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Action
    retry_date: date | None = None
    retry_time: time = DEFAULT_PRESENTMENT_TIME
    reasoning: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _retry_date_iff_retry(self) -> "RetryDecision":
        if self.action == "RETRY" and self.retry_date is None:
            raise ValueError("RETRY requires a retry_date")
        if self.action != "RETRY" and self.retry_date is not None:
            raise ValueError(f"{self.action} must not carry a retry_date")
        return self

    @property
    def is_terminal(self) -> bool:
        return self.action in TERMINAL_ACTIONS


class PolicyState(BaseModel):
    """Everything a policy is allowed to know, on one day of one cycle.

    Constructible only from ``agent_view`` output. Note what is absent: the
    balance series, the salary day, the opt-out sensitivity, and even
    ``balance_at_failure`` - a real PSP learns the account was short, never by
    how much.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    customer: AgentCustomer
    payment_history: tuple[AgentPaymentHistoryRecord, ...]
    failure_event: AgentFailureEvent
    today: date
    attempts_used: int = Field(ge=0)
    notifications_sent_this_cycle: int = Field(ge=0)
    cycle_end_date: date

    @property
    def days_since_failure(self) -> int:
        return (self.today - self.failure_event.original_debit_date).days

    @property
    def attempts_remaining(self) -> int:
        """Presentments still permitted this cycle under the NPCI ceiling."""
        return rules.max_retries_remaining(self.attempts_used)

    @property
    def earliest_compliant_retry_date(self) -> date:
        """The soonest date a retry may be scheduled for.

        RBI Sec 6(a): the issuer must notify at least 24 hours ahead, so a
        decision made today can only commit to tomorrow at the earliest.
        """
        from datetime import timedelta

        return self.today + timedelta(days=1)

    @property
    def days_left_in_cycle(self) -> int:
        return (self.cycle_end_date - self.today).days


class Policy(ABC):
    """Base class for every retry policy."""

    name: str = "unnamed"

    @abstractmethod
    def decide(self, state: PolicyState) -> RetryDecision:
        """Choose an action for ``state.today``.

        The simulator consults a policy only when no retry is already pending,
        so a policy never has to remember what it scheduled: everything it
        needs is derivable from ``attempts_used`` and the history.
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"<{type(self).__name__} name={self.name!r}>"
