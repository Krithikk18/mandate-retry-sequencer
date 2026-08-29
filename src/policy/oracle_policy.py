"""=========================  THIS POLICY CHEATS  =========================

OraclePolicy reads the hidden daily balance series. It is NOT a policy anyone
could deploy: it knows, on the day of the failure, exactly which future date
the customer's salary will land and whether it will clear the mandate. No real
sequencer can know that, because the balance on a future date does not exist
yet and the bank would not share it if it did.

It exists for one reason: to put a NUMBER on the ceiling. "Our agent recovered
62%" means nothing on its own. "Our agent recovered 62% of a possible 78%,
against a fixed-schedule floor of 37%" is a claim with a scale attached.

It is deliberately NOT exported from ``policy/__init__.py``. Import it by its
full path so the cheat is visible at the call site:

    from policy.oracle_policy import OraclePolicy

A test asserts that no other module under ``policy/`` imports it, and that no
agent-safe policy imports ``generator.oracle`` at all.

The one thing it does NOT get is a free pass on the rules. It obeys the same
NPCI attempt ceiling, the same 24-hour notice lead time and the same execution
window as every other policy - otherwise the ceiling would be unreachable by
construction and useless as a target.
========================================================================
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from datetime import datetime

from generator import rules
from generator.banks import BankOutageBook
from generator.models import Customer
from generator.oracle import first_success_date, would_succeed_at

from .base import Policy, PolicyState, RetryDecision


class OraclePolicy(Policy):
    """Schedules the earliest date that will actually clear. The ceiling.

    Given the truth, the optimal play under this cost model is simple:

    * if some reachable date clears, present on the earliest such date - one
      presentment, one notification, minimum opt-out exposure;
    * if no date clears, HOLD. Never CANCEL: the mandate is structurally alive,
      the customer never chose to leave, and holding costs nothing while
      cancelling forfeits the whole annual LTV.

    That second branch is where most of the gap over a fixed schedule comes
    from, and it is worth being explicit that it is a consequence of the cost
    model rather than a clever trick.
    """

    name = "oracle_ceiling"

    def __init__(
        self,
        customers: dict[str, Customer],
        balance_series: dict[str, pd.Series],
        horizon_days: int = 30,
        outages: BankOutageBook | None = None,
    ) -> None:
        self._customers = customers
        self._balances = balance_series
        self._outages = outages
        self.horizon_days = horizon_days

    @classmethod
    def for_dataset(cls, dataset, horizon_days: int = 30) -> "OraclePolicy":
        """Bind to ONE dataset's truth.

        Customer ids are per-dataset, not global: ``CUST_0001`` is a different
        person in every seed. Merging several datasets' truth tables into one
        dict silently overwrites, and the oracle then reads the wrong person's
        balance while still looking like an oracle. Always bind per seed.
        """
        from generator.generate import outage_book_for

        return cls(
            {c.customer_id: c for c in dataset.customers},
            dataset.balance_series,
            horizon_days=horizon_days,
            outages=outage_book_for(dataset),
        )

    def decide(self, state: PolicyState) -> RetryDecision:
        if state.failure_event.failure_class in rules.NEVER_RETRY_CLASSES:
            # The oracle cheats on information, never on the rules. An expired
            # credential and a revoked mandate cannot clear on any date, and
            # knowing the future does not change that.
            return RetryDecision(
                action="CANCEL"
                if state.failure_event.failure_class in rules.RETRY_IS_BREACH_CLASSES
                else "HOLD",
                reasoning={
                    "policy": self.name,
                    "rule": f"{state.failure_event.failure_class}: no date can succeed",
                    "cheats": True,
                },
            )

        customer_id = state.customer.customer_id
        truth = self._customers[customer_id]
        series = self._balances[customer_id]
        amount = state.customer.subscription_amount

        # Guard against being bound to the wrong dataset. Ids repeat across
        # seeds, so a mis-bound oracle would quietly grade itself against
        # somebody else's bank balance and simply score badly, which is far
        # worse than crashing.
        if truth.subscription_amount != amount or truth.rail != state.customer.rail:
            raise ValueError(
                f"{self.name} is bound to the wrong dataset: {customer_id} has "
                f"amount {truth.subscription_amount}/{truth.rail} in the truth "
                f"table but {amount}/{state.customer.rail} in the policy state. "
                "Build one OraclePolicy per seed with OraclePolicy.for_dataset."
            )

        if state.attempts_remaining <= 0:
            return RetryDecision(
                action="HOLD",
                reasoning={
                    "policy": self.name,
                    "rule": "attempt budget spent; hold rather than cancel",
                },
            )

        # Look only from the first date this decision could legally reach.
        earliest = state.earliest_compliant_retry_date
        if self._outages is not None:
            # Bank-aware: a date whose balance is fine but whose bank is down
            # would fail, and a ceiling that schedules into an outage is not a
            # ceiling. Walk forward to the first date that clears BOTH.
            target = None
            day = earliest
            while day <= state.cycle_end_date:
                at = datetime.combine(day, rules.DEFAULT_PRESENTMENT_TIME)
                if would_succeed_at(truth, at, amount, series, self._outages):
                    target = day
                    break
                day += timedelta(days=1)
        else:
            target = first_success_date(
                truth,
                series,
                amount,
                after=earliest - timedelta(days=1),
                horizon_days=self.horizon_days,
            )

        if target is not None and target <= state.cycle_end_date:
            return RetryDecision(
                action="RETRY",
                retry_date=target,
                reasoning={
                    "policy": self.name,
                    "rule": "earliest date the balance actually clears",
                    "cheats": True,
                    "days_after_failure": (
                        target - state.failure_event.original_debit_date
                    ).days,
                },
            )

        return RetryDecision(
            action="HOLD",
            reasoning={
                "policy": self.name,
                "rule": "no reachable date clears; keep the mandate alive",
                "cheats": True,
                "horizon_days": self.horizon_days,
                "first_success_date": None if target is None else target.isoformat(),
            },
        )
