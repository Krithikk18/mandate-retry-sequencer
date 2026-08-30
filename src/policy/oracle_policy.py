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
from generator.afa import MAX_USEFUL_NUDGES
from generator.banks import BankOutageBook
from generator.models import Customer
from generator.oracle import first_success_date, would_succeed_at

from .base import Policy, PolicyState, RetryDecision


class OraclePolicy(Policy):
    """Schedules the earliest date that will actually clear. The ceiling.

    Given the truth, the optimal play under this cost model is simple:

    * if a nudge would bring the earliest clearing date forward, send one
      first - see below;
    * if some reachable date clears, present on the earliest such date - one
      presentment, one notification, minimum opt-out exposure;
    * if no date clears, HOLD. Never CANCEL: the mandate is structurally alive,
      the customer never chose to leave, and holding costs nothing while
      cancelling forfeits the whole annual LTV.

    That third branch is where most of the gap over a fixed schedule comes
    from, and it is worth being explicit that it is a consequence of the cost
    model rather than a clever trick.

    **Why the ceiling nudges.** Through class 4 the oracle only ever chose a
    date, because the truth was fixed and nothing a policy did could move it.
    Class 5 is not like that: above the RBI Sec 8(a) ceiling a debit clears
    only once the customer authenticates, and a nudge causally raises the
    chance they have. An oracle that refused to nudge would be an upper bound
    over a SMALLER action set than the agent's, so the agent could legitimately
    beat it - and "the agent never exceeds the ceiling" is one of the load-
    bearing tests in this project. The ceiling is capped at
    ``afa.MAX_USEFUL_NUDGES`` for the opposite reason: an oracle allowed
    unlimited messages would be unattainable, which makes every real policy
    look worse than it is.

    This is the same failure mode class 2 hit from the other side, where the
    oracle scheduled into live outages because it used a balance-only lookup.
    Both times the fix was to make the ceiling answer the question the agent is
    actually being scored on.
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
        nudges = state.nudges_sent_this_cycle
        target = self._first_clearing_date(
            truth, series, amount, state, earliest, nudges
        )

        # Would one more nudge do better? One-step lookahead, and it has to
        # account for the day the nudge costs: deciding to nudge today means
        # the next decision is tomorrow, so the soonest presentment afterwards
        # is the day after that. The FIRST nudge is usually free in time
        # anyway, because the 24-hour notice already rules out presenting
        # today. Only taken when it strictly improves - an oracle that spends a
        # day on a mandate that was already fine would not be a ceiling.
        if truth.requires_otp and nudges < MAX_USEFUL_NUDGES:
            after_nudge = self._first_clearing_date(
                truth,
                series,
                amount,
                state,
                earliest + timedelta(days=1),
                nudges + 1,
            )
            if after_nudge is not None and (target is None or after_nudge < target):
                return RetryDecision(
                    action="NUDGE",
                    reasoning={
                        "policy": self.name,
                        "rule": "authentication is the binding gate; ask for it",
                        "cheats": True,
                        "nudges_sent": nudges,
                        "clears_on_without_nudge": None if target is None else target.isoformat(),
                        "clears_on_with_nudge": after_nudge.isoformat(),
                    },
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

    def _first_clearing_date(
        self,
        truth: Customer,
        series,
        amount: float,
        state: PolicyState,
        earliest,
        nudges: int,
    ):
        """Earliest date from ``earliest`` that clears EVERY gate, or None.

        Bank-aware when an outage book is bound: a date whose balance is fine
        but whose bank is down would fail, and a ceiling that schedules into an
        outage is not a ceiling.
        """
        if earliest > state.cycle_end_date:
            return None
        if self._outages is not None:
            day = earliest
            while day <= state.cycle_end_date:
                at = datetime.combine(day, rules.DEFAULT_PRESENTMENT_TIME)
                if would_succeed_at(
                    truth, at, amount, series, self._outages, nudges=nudges
                ):
                    return day
                day += timedelta(days=1)
            return None
        return first_success_date(
            truth,
            series,
            amount,
            after=earliest - timedelta(days=1),
            horizon_days=self.horizon_days,
            nudges=nudges,
        )
