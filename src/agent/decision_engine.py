"""The money equation: what is each candidate retry date actually worth?

For every date the policy could legally present on:

    value(d) = P_clear(d) * amount                     expected recovery
             - processing_cost[rail]                   cost of presenting
             - (1 - P_clear(d)) * bounce_charge[rail]  expected bounce charge
             - P_optout(...) * annual_ltv              expected churn

``P_clear`` comes from the tier-2 payday model. ``P_optout`` comes from the
agent's own churn estimator, which sees tenure and notification count and
nothing else - the true hazard reads a hidden per-customer sensitivity and is
not callable from here.

The engine returns the WHOLE scored list, not the argmax. The runners-up are
what make a decision auditable: "we chose the 7th" is an assertion, while "we
chose the 7th at +Rs 210 over the 4th at -Rs 38, because the 4th was five days
before payday" is a reason.

Two things this equation makes visible that a fixed schedule cannot see:

* The bounce term is rail-dependent. A NACH retry with a 20% chance of
  clearing carries an 80% chance of putting Rs 500 on the customer's bank
  statement, so the same date is worth presenting on UPI and not on NACH.
* The churn term scales with annual LTV, not with the amount at stake. Chasing
  Rs 199 through four notifications risks Rs 2,388 of subscription value, and
  the equation prices that rather than assuming retries are free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time, timedelta
from typing import Any, Mapping

from generator import rules
from generator.models import AgentCustomer
from policy.base import DEFAULT_PRESENTMENT_TIME, PolicyState

from .churn import (
    DEFAULT_IRRITATION_DIRECTION,
    IrritationDirection,
    estimated_optout_probability,
)

# --- NUDGE ------------------------------------------------------------------
# A nudge is an out-of-band message ("your payment failed, top up before the
# 7th"). It is not a pre-debit notification, so it consumes no attempt and
# creates no regulated opt-out - but it does spend goodwill.
#
# BOTH ESTIMATES, and deliberately small. There is no public figure for how
# often a dunning email causes a customer to fund an account, and this is not
# the contribution of the project. If the ablation shows NUDGE carrying real
# weight, that is a signal the numbers are too generous, not a finding.
NUDGE_CLEAR_LIFT: float = 0.05  # additive bump to P_clear on the target date
NUDGE_GOODWILL_COST_FRACTION: float = 0.002  # of annual LTV, per nudge
NUDGE_MIN_P_CLEAR: float = 0.25  # below this a nudge cannot rescue the date
NUDGE_MAX_P_CLEAR: float = 0.80  # above this the date needs no help


@dataclass(frozen=True)
class ScoredDate:
    """One candidate presentment date, fully priced."""

    day: date
    p_clear: float
    p_optout: float
    expected_recovery: float
    processing_cost: float
    expected_bounce: float
    expected_churn: float
    value: float
    days_after_failure: int

    def as_dict(self) -> dict[str, Any]:
        """Compact form for the audit trail."""
        return {
            "date": self.day.isoformat(),
            "days_after_failure": self.days_after_failure,
            "p_clear": round(self.p_clear, 4),
            "p_optout": round(self.p_optout, 4),
            "expected_recovery": round(self.expected_recovery, 2),
            "processing_cost": round(self.processing_cost, 2),
            "expected_bounce": round(self.expected_bounce, 2),
            "expected_churn": round(self.expected_churn, 2),
            "value": round(self.value, 2),
        }


@dataclass(frozen=True)
class Constraint:
    """Why a date was not available. Kept for the audit trail."""

    reason: str
    detail: str = ""


@dataclass
class DecisionEngine:
    """Prices candidate dates. Decides nothing - see sequencer.py."""

    # Hours the issuer's pre-debit notice must precede the debit. 24 is the RBI
    # requirement; the lead-time ablation varies it to price the regulation.
    lead_time_hours: int = rules.PRE_DEBIT_NOTICE_HOURS
    irritation_direction: IrritationDirection = DEFAULT_IRRITATION_DIRECTION
    presentment_time: time = DEFAULT_PRESENTMENT_TIME
    include_optout_cost: bool = True  # ablation switch
    # Rupees collected sooner are worth more than rupees collected later, and
    # the equation as written says otherwise: it will happily wait a month to
    # shave the churn term, because nothing penalises delay.
    #
    # DEFAULT 0.0 - the equation is unchanged unless a caller asks. Turning it
    # on is what `scripts/run_downtime.py` measures, because the downtime class
    # is where the omission actually shows: the money was collectable the next
    # morning and the agent waited weeks for a marginally safer notice.
    daily_discount: float = 0.0
    _binding: list[Constraint] = field(default_factory=list, init=False)

    @property
    def lead_time_days(self) -> int:
        """Lead time in whole days, which is the simulator's resolution.

        0h means an agent that could react to a salary credit the same day.
        Anything from 1 to 24 hours rounds to one day, so 6h/12h/24h are
        indistinguishable here - the ablation reports that honestly rather than
        manufacturing a difference.
        """
        return -(-self.lead_time_hours // 24)  # ceiling division

    def earliest_presentment_date(self, today: date) -> date:
        return today + timedelta(days=self.lead_time_days)

    def candidate_dates(self, state: PolicyState) -> list[date]:
        """Every date this decision could legally commit to.

        Bounded by the notice period, the end of the billing cycle, and the
        NPCI attempt budget. A policy that proposes anything outside this set
        raises a ComplianceViolation in the simulator rather than being
        quietly clamped, so the set is the contract.
        """
        self._binding = []
        if state.attempts_remaining <= 0:
            self._binding.append(
                Constraint("attempt_budget", f"{state.attempts_used} presentments used")
            )
            return []
        if not rules.in_execution_window(self.presentment_time, state.customer.rail):
            self._binding.append(Constraint("execution_window", str(self.presentment_time)))
            return []

        first = self.earliest_presentment_date(state.today)
        if first > state.cycle_end_date:
            self._binding.append(
                Constraint(
                    "cycle_end",
                    f"earliest presentment {first} is past cycle end {state.cycle_end_date}",
                )
            )
            return []

        days = (state.cycle_end_date - first).days
        return [first + timedelta(days=offset) for offset in range(days + 1)]

    def binding_constraints(self) -> list[dict[str, str]]:
        return [{"reason": c.reason, "detail": c.detail} for c in self._binding]

    def score(
        self,
        state: PolicyState,
        liquidity: Mapping[date, float],
        nudge: bool = False,
    ) -> list[ScoredDate]:
        """Price every candidate date, best first.

        ``liquidity`` is the payday model's curve. ``nudge`` applies the
        documented lift, for comparing "nudge then present" against "present".
        """
        customer = state.customer
        amount = customer.subscription_amount
        processing = rules.processing_cost(customer.rail)
        bounce = rules.bounce_charge(customer.rail)
        original = state.failure_event.original_debit_date

        scored: list[ScoredDate] = []
        for day in self.candidate_dates(state):
            p_clear = float(liquidity.get(day, 0.0))
            if nudge:
                p_clear = min(1.0, p_clear + NUDGE_CLEAR_LIFT)

            # The customer reacts to the NOTICE, not to the debit, and the
            # notice goes out `lead_time_days` earlier. A longer lead therefore
            # lands the message closer to the fresh bounce, which under the
            # default irritation direction makes it more likely to be acted on.
            notice_day = day - timedelta(days=self.lead_time_days)
            days_since_failure = max(0, (notice_day - original).days)

            p_optout = (
                estimated_optout_probability(
                    customer,
                    state.notifications_sent_this_cycle + 1,
                    days_since_failure,
                    self.irritation_direction,
                )
                if self.include_optout_cost
                else 0.0
            )

            expected_recovery = p_clear * amount
            if self.daily_discount:
                days_out = max(0, (day - original).days)
                expected_recovery *= (1.0 - self.daily_discount) ** days_out
            expected_bounce = (1.0 - p_clear) * bounce
            expected_churn = p_optout * customer.annual_ltv
            value = expected_recovery - processing - expected_bounce - expected_churn

            scored.append(
                ScoredDate(
                    day=day,
                    p_clear=p_clear,
                    p_optout=p_optout,
                    expected_recovery=expected_recovery,
                    processing_cost=processing,
                    expected_bounce=expected_bounce,
                    expected_churn=expected_churn,
                    value=value,
                    days_after_failure=(day - original).days,
                )
            )

        scored.sort(key=lambda s: (-s.value, s.day))
        return scored

    def nudge_value(self, state: PolicyState, liquidity: Mapping[date, float]) -> float:
        """Gain from nudging before presenting, net of the goodwill it spends.

        Deliberately modest: a nudge only helps where the date is already
        plausible but not certain. Outside that band it is a message sent for
        the sake of sending one.
        """
        plain = self.score(state, liquidity)
        if not plain:
            return 0.0
        best_plain = plain[0]
        if not (NUDGE_MIN_P_CLEAR <= best_plain.p_clear <= NUDGE_MAX_P_CLEAR):
            return 0.0
        boosted = self.score(state, liquidity, nudge=True)
        cost = NUDGE_GOODWILL_COST_FRACTION * state.customer.annual_ltv
        return boosted[0].value - best_plain.value - cost


def expected_value_of_waiting(scored: list[ScoredDate], today_best: ScoredDate | None) -> bool:
    """True when the best date is far enough out that committing now is premature.

    Committing early is not free: the notice fires `lead_time_days` before the
    debit, and every day of waiting is a day of fresher evidence. If the best
    date is not reachable from today's earliest slot, there is nothing to gain
    by naming it now.
    """
    if not scored or today_best is None:
        return False
    return scored[0].day > today_best.day
