"""Runs one policy over one failure event, cycle by cycle and day by day.

The simulator is the only place where policy decisions meet the truth.

**An episode spans cycles, not days.** HOLD is not a terminal action: it means
"skip this cycle, keep the mandate alive, re-present next cycle". So the
simulator carries the mandate forward, the mandate auto-presents on its anchor
date in every subsequent cycle, and the episode ends only when the money
arrives, the mandate dies, or the horizon runs out. That is what makes holding
cost something - and therefore what makes a stopping rule capable of being
wrong.

Within a cycle, on each day:

1. resolve any presentment that comes due today, and
2. consult the policy if nothing is pending.

**The policy is consulted only when no retry is already scheduled.** That is
the protocol that lets a policy be stateless: it never has to remember what it
committed to, because it is only asked when there is nothing outstanding.

Compliance violations RAISE. A policy that schedules a fifth presentment in a
cycle, or one inside the 24-hour notice window, or outside the execution
window, fails loudly. Silently clamping the date would turn an illegal policy
into a legal-looking one with a good number attached, which is the single most
dangerous failure mode an evaluation harness can have.

**NUDGE is the one action that changes the world rather than selecting from
it.** Every other action picks a date or declines to; the truth underneath is
fixed and the policy can only estimate it. A nudge is an out-of-band message
that causally raises the chance the customer completes AFA for the next
presentment, so the simulator has to apply it to the truth rather than log it
and move on. Three properties keep it honest, and each is pinned by a test:

* it is NOT a pre-debit notification, so it neither increments the notification
  counter nor creates an RBI Sec 6(c) opt-out opportunity - that distinction is
  the entire reason nudging is cheaper than presenting;
* it consumes no NPCI attempt;
* it is not free - ``rules.NUDGE_GOODWILL_COST_FRACTION`` of annual LTV comes
  off net value per nudge, and the lift has diminishing returns that converge
  below certainty (see ``generator.afa.completion_probability``).

The effect is deliberately CLASS-AGNOSTIC here. A nudge raises OTP-completion
odds for mandates that need OTP and does nothing for a customer whose problem
is an empty account or a bank that is down. The simulator never asks which
failure class it is looking at; the arithmetic in ``generator.afa`` does the
discriminating, so the policy shape stays a consequence rather than a branch.

Both random draws are keyed on identity, never on a shared stream:

* opt-out    -> (customer_id, notification_index, seed)
* attrition  -> (customer_id, cycle_index, seed)
* AFA        -> (balance_seed, presentment_date)   [generator.afa]

Off a shared generator, a policy that made three attempts would shift the
stream for the next policy and the comparison would measure RNG alignment
rather than skill. Common random numbers also cut the variance of the
comparison itself.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, time, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd

from generator import rules
from generator.models import (
    AgentCustomer,
    AgentFailureEvent,
    AgentPaymentHistoryRecord,
    Customer,
    FailureEvent,
)
from generator.optout import (
    DEFAULT_IRRITATION_DIRECTION,
    IrritationDirection,
    draw_optout,
)
from generator.ambiguous import AmbiguityConfig
from generator.oracle import resolve_presentment
from policy.base import Policy, PolicyState

from . import economics

Outcome = Literal["RECOVERED", "HELD", "CANCELLED", "OPTED_OUT", "CHURNED"]


class ComplianceViolation(Exception):
    """A policy proposed something the regulations do not permit."""


@dataclass(frozen=True)
class DecisionLogEntry:
    """One day of the episode, as it would appear in an audit trail.

    ``attempts_used`` and ``notifications_sent`` are SNAPSHOTS of the PER-CYCLE
    counters, taken when the entry was written - the NPCI ceiling is a
    per-cycle rule, so those are the numbers a reviewer needs. The audit trail reads them
    directly rather than replaying the episode to work out what the counters
    were at each step - a rendered trail that recomputes its own numbers can
    drift from what actually happened, which defeats the point of having one.
    """

    day: date
    action: str
    retry_date: date | None
    reasoning: dict[str, Any]
    event: str = ""
    cycle: int = 1
    attempts_used: int = 0
    notifications_sent: int = 0
    retry_time: time | None = None

    def render(self) -> str:
        target = f" -> {self.retry_date}" if self.retry_date else ""
        note = f"  [{self.event}]" if self.event else ""
        return f"c{self.cycle} {self.day}  {self.action:<7}{target}{note}"


@dataclass
class EpisodeResult:
    """What one policy achieved on one failure event.

    ``customer`` is the AGENT-facing record, not the truth one: an audit trail
    is something you hand to a reviewer, and it must not carry fields the
    policy could not see either.
    """

    customer_id: str
    policy_name: str
    outcome: Outcome
    recovered_amount: float
    attempts_used: int
    notifications_sent: int
    bounce_charges_paid: float
    carrying_cost_paid: float
    cycles_held: int
    compliance_breaches: int
    ltv_retained: float
    ltv_lost: float
    net_value: float
    days_to_resolution: int
    decision_log: list[DecisionLogEntry] = field(default_factory=list)
    retry_dates: tuple[date, ...] = ()
    customer: AgentCustomer | None = None
    failure_class: str = rules.FAILURE_CLASS_INSUFFICIENT_FUNDS
    return_code: str = rules.RETURN_CODE_INSUFFICIENT_FUNDS
    original_debit_date: date | None = None

    # Out-of-band nudges sent, what they cost in goodwill, and how many
    # presentments cleared AFA that would NOT have cleared unnudged. The last
    # one is an exact counterfactual rather than an estimate: the AFA draw is
    # keyed on identity and date only, so re-testing the same uniform against
    # the unnudged probability answers "would this have happened anyway?".
    nudges_sent: int = 0
    goodwill_cost_paid: float = 0.0
    otp_completions_from_nudges: int = 0

    # Presentments refused at the authentication gate. THE class-5 waste
    # number: an attempt spent, an RBI Sec 6(a) notice fired, a bounce charge
    # risked on NACH, and nothing learned - the debit never reached the
    # account, so it did not even establish that the customer was short. A
    # fixed schedule cannot avoid these, because it never reads the return code
    # and so never discovers an authentication is outstanding.
    presentments_refused_for_afa: int = 0

    @property
    def recovered(self) -> bool:
        return self.outcome == "RECOVERED"

    @property
    def mandate_survives(self) -> bool:
        return economics.mandate_survives(self.outcome)


def cycle_end_date(original_debit_date: date) -> date:
    """Last day of the billing cycle a debit belongs to.

    A monthly mandate runs debit-date to debit-date, so the cycle ends the day
    before the next presentment falls due. Debit days are capped at 28, so the
    next month always has the day.
    """
    return next_debit_date(original_debit_date) - timedelta(days=1)


def next_debit_date(debit_date: date) -> date:
    """Same day, next month."""
    year, month = debit_date.year, debit_date.month
    year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return date(year, month, debit_date.day)


def _rng_from(*parts: object) -> np.random.Generator:
    key = "|".join(str(p) for p in parts).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def optout_rng(customer_id: str, notification_index: int, seed: int) -> np.random.Generator:
    """A generator determined solely by who, which notice, and which seed.

    Deliberately independent of the policy: two policies that put their Nth
    notification on the same customer draw the same uniform, so any difference
    in outcome is a difference in the probability they created, not noise.
    """
    return _rng_from(customer_id, notification_index, seed)


def attrition_rng(customer_id: str, cycle_index: int, seed: int) -> np.random.Generator:
    """Natural churn on a held customer. A separate namespace from the opt-out
    draw, so the two can never collide on the same key."""
    return _rng_from(customer_id, "attrition", cycle_index, seed)


def draws_natural_attrition(customer_id: str, cycle_index: int, seed: int) -> bool:
    """Did this held customer drift away of their own accord this cycle?"""
    return bool(
        attrition_rng(customer_id, cycle_index, seed).random()
        < rules.NATURAL_ATTRITION_PER_CYCLE
    )


def _check_compliance(
    decision_date: date,
    retry_date: date,
    retry_time,
    attempts_used: int,
    rail: str,
    policy_name: str,
    lead_time_days: int = 1,
    failure_class: str = rules.FAILURE_CLASS_INSUFFICIENT_FUNDS,
) -> None:
    """Raise unless the proposed presentment is legal. Never clamp.

    ``lead_time_days`` is the notice period. One day is the RBI requirement;
    scripts/lead_time_ablation.py varies it to price the regulation, which is
    the only reason it is a parameter rather than a constant.
    """
    if failure_class in rules.RETRY_IS_BREACH_CLASSES:
        # Not an efficiency question. The customer withdrew consent at a
        # pre-debit notice under RBI Sec 6(c), so there is no mandate left to
        # present against. This raises rather than scoring badly, because a
        # harness that let it through would be reporting a number obtained by
        # breaking the law. Retrying an EXPIRED CARD is futile but lawful, and
        # is counted as waste instead.
        raise ComplianceViolation(
            f"{policy_name}: presented against a {failure_class} mandate on "
            f"{retry_date}. Consent was withdrawn under RBI Sec 6(c); this "
            "mandate must never be presented again."
        )

    earliest = decision_date + timedelta(days=lead_time_days)
    if retry_date < earliest:
        raise ComplianceViolation(
            f"{policy_name}: retry on {retry_date} decided on {decision_date} "
            f"breaks the RBI 24-hour pre-debit notice (earliest is {earliest})"
        )
    if attempts_used >= rules.MAX_ATTEMPTS_PER_CYCLE:
        raise ComplianceViolation(
            f"{policy_name}: presentment {attempts_used + 1} exceeds the NPCI "
            f"ceiling of {rules.MAX_ATTEMPTS_PER_CYCLE} attempts per cycle"
        )
    if not rules.in_execution_window(retry_time, rail):
        raise ComplianceViolation(
            f"{policy_name}: presentment at {retry_time} on {rail} falls outside "
            f"the {rules.UPI_AUTOPAY_WINDOW_START}-{rules.UPI_AUTOPAY_WINDOW_END} "
            "execution window"
        )


def _stamped(
    ep: "_Episode",
    day: date,
    action: str,
    retry_date: date | None,
    reasoning: dict[str, Any],
    event: str = "",
    cycle: int = 1,
    retry_time: time | None = None,
) -> DecisionLogEntry:
    """Build a log entry stamped with the counters as they stand right now."""
    return DecisionLogEntry(
        day=day,
        action=action,
        retry_date=retry_date,
        reasoning=reasoning,
        event=event,
        cycle=cycle,
        attempts_used=ep.cycle_attempts,
        notifications_sent=ep.cycle_notifications,
        retry_time=retry_time,
    )


def _history_up_to(
    payment_history: list[AgentPaymentHistoryRecord], customer_id: str, day: date
) -> list[AgentPaymentHistoryRecord]:
    return [
        r for r in payment_history if r.customer_id == customer_id and r.attempt_date <= day
    ]


@dataclass
class _Episode:
    """Mutable accumulator threaded through the cycle loop."""

    attempts_used: int = 0
    notifications: int = 0
    # Per-cycle counters, kept alongside the episode totals because the NPCI
    # ceiling is a per-cycle rule and the audit trail has to show it that way.
    cycle_attempts: int = 0
    cycle_notifications: int = 0
    # Nudges are counted PER CYCLE as well as per episode. A nudge asks the
    # customer to authorise the debit that is coming, and AFA is required per
    # transaction, so the ask does not carry into next month's debit - the
    # counter resets with the cycle exactly as the attempt counter does.
    nudges: int = 0
    cycle_nudges: int = 0
    goodwill_cost: float = 0.0
    otp_completions_from_nudges: int = 0
    afa_refused: int = 0
    failed_presentments: int = 0
    cycles_held: int = 0
    compliance_breaches: int = 0
    recovered_amount: float = 0.0
    outcome: Outcome | None = None
    resolved_on: date | None = None
    log: list[DecisionLogEntry] = field(default_factory=list)
    retry_dates: list[date] = field(default_factory=list)
    history: list[AgentPaymentHistoryRecord] = field(default_factory=list)


def run_episode(
    policy: Policy,
    customer: Customer,
    failure_event: FailureEvent,
    balance_series: pd.Series,
    payment_history: list[AgentPaymentHistoryRecord],
    seed: int,
    irritation_direction: IrritationDirection = DEFAULT_IRRITATION_DIRECTION,
    max_cycles: int = rules.MAX_EPISODE_CYCLES,
    lead_time_days: int = 1,
    notifications_can_cancel: bool = True,
    strict_compliance: bool = True,
    ambiguity: AmbiguityConfig | None = None,
) -> EpisodeResult:
    """Play ``policy`` against one failure event and score it against truth.

    ``customer`` and ``balance_series`` are the hidden truth and are used ONLY
    to resolve outcomes. What the policy sees is rebuilt each day from the
    agent-facing records.
    """
    agent_customer = AgentCustomer.from_customer(customer)
    amount = customer.subscription_amount
    first_failure = failure_event.original_debit_date
    series_end = balance_series.index[-1].date()

    ep = _Episode(history=_history_up_to(payment_history, customer.customer_id, first_failure))
    ep.resolved_on = first_failure

    cycle_debit = first_failure
    for cycle_index in range(max_cycles):
        cycle_end = cycle_end_date(cycle_debit)
        if cycle_end > series_end:
            # The simulated world stops here; report what is known so far
            # rather than extrapolating past the truth.
            break

        # Cycle 1's original presentment already happened - it is the failure
        # event we were handed. Later cycles auto-present on the anchor date,
        # because a live mandate keeps billing whether or not anyone decided so.
        # A mandate held on a structurally dead class is SUSPENDED, not
        # re-presented. Nobody knowingly auto-bills an expired credential every
        # month; billing is paused until the customer fixes it. Without this,
        # holding such a mandate would look more expensive than cancelling it
        # purely because of presentments a real merchant would never make.
        dead_class = failure_event.failure_class in rules.NEVER_RETRY_CLASSES
        if cycle_index > 0 and dead_class:
            ep.cycles_held += 1
            ep.resolved_on = cycle_end
            if draws_natural_attrition(customer.customer_id, cycle_index, seed):
                ep.outcome, ep.resolved_on = "CHURNED", cycle_end
                break
            cycle_debit = next_debit_date(cycle_debit)
            continue

        ep.attempts_used += 1
        cycle_attempts = 1
        ep.cycle_attempts = 1
        ep.cycle_notifications = 0
        # AFA is required per transaction, so last cycle's authorisation does
        # not carry. Reset before the auto-presentment resolves, since that
        # presentment is the new cycle's first and has not been nudged for.
        ep.cycle_nudges = 0
        if cycle_index == 0:
            ep.failed_presentments += 1
        else:
            cleared, code = _record_presentment(
                ep, customer, cycle_debit, amount, balance_series, ambiguity
            )
            ep.log.append(
                _stamped(ep,
                    cycle_debit,
                    "AUTO",
                    cycle_debit,
                    {"rule": "mandate re-presents on its anchor date"},
                    "debit cleared" if cleared else f"debit failed, code {code}",
                    cycle_index + 1,
                )
            )
            if cleared:
                ep.outcome, ep.recovered_amount, ep.resolved_on = (
                    "RECOVERED",
                    amount,
                    cycle_debit,
                )
                break

        terminal = _run_cycle(
            policy=policy,
            ep=ep,
            customer=customer,
            agent_customer=agent_customer,
            cycle_debit=cycle_debit,
            cycle_end=cycle_end,
            cycle_index=cycle_index,
            cycle_attempts=cycle_attempts,
            amount=amount,
            balance_series=balance_series,
            seed=seed,
            irritation_direction=irritation_direction,
            lead_time_days=lead_time_days,
            notifications_can_cancel=notifications_can_cancel,
            failure_class=failure_event.failure_class,
            return_code=failure_event.return_code,
            strict_compliance=strict_compliance,
            ambiguity=ambiguity,
        )
        if terminal:
            break

        # The cycle ended without resolution: the mandate is carried.
        ep.cycles_held += 1
        ep.resolved_on = cycle_end
        if draws_natural_attrition(customer.customer_id, cycle_index, seed):
            ep.outcome, ep.resolved_on = "CHURNED", cycle_end
            ep.log.append(
                _stamped(ep,
                    cycle_end,
                    "CHURN",
                    None,
                    {"rule": "held customer left of their own accord"},
                    "natural attrition",
                    cycle_index + 1,
                )
            )
            break

        cycle_debit = next_debit_date(cycle_debit)

    outcome: Outcome = ep.outcome or "HELD"
    bounce = ep.failed_presentments * rules.bounce_charge(customer.rail)
    carrying = economics.carrying_cost_total(amount, ep.cycles_held)
    goodwill = ep.goodwill_cost

    return EpisodeResult(
        customer_id=customer.customer_id,
        policy_name=policy.name,
        outcome=outcome,
        recovered_amount=ep.recovered_amount,
        attempts_used=ep.attempts_used,
        notifications_sent=ep.notifications,
        bounce_charges_paid=bounce,
        carrying_cost_paid=carrying,
        cycles_held=ep.cycles_held,
        compliance_breaches=ep.compliance_breaches,
        ltv_retained=economics.ltv_retained(agent_customer, outcome),
        ltv_lost=economics.ltv_lost(agent_customer, outcome),
        net_value=economics.net_value(
            agent_customer,
            outcome,
            ep.recovered_amount,
            ep.attempts_used,
            bounce,
            ep.cycles_held,
            goodwill,
        ),
        days_to_resolution=((ep.resolved_on or first_failure) - first_failure).days,
        decision_log=ep.log,
        retry_dates=tuple(ep.retry_dates),
        customer=agent_customer,
        failure_class=failure_event.failure_class,
        return_code=failure_event.return_code,
        original_debit_date=first_failure,
        nudges_sent=ep.nudges,
        goodwill_cost_paid=goodwill,
        otp_completions_from_nudges=ep.otp_completions_from_nudges,
        presentments_refused_for_afa=ep.afa_refused,
    )


def _record_presentment(
    ep: _Episode,
    customer: Customer,
    day: date,
    amount: float,
    balance_series: pd.Series,
    ambiguity: AmbiguityConfig | None = None,
) -> tuple[bool, str | None]:
    """Present the debit, append it to history, report (cleared, return_code).

    The return code matters as much as the boolean now. It is the only channel
    the agent has, so writing 04 onto a decline that was really an
    authentication refusal would teach the payday model the account was empty -
    the same bug class 2 surfaced with code 91, and the reason
    ``resolve_presentment`` is the single place that decides.

    Nudge attribution is measured here because here is where the truth is
    consulted. A completion counts as caused by the nudges only if the SAME
    draw would have failed with none of them: the AFA uniform is keyed on
    identity and date, never on the nudge count, so the unnudged branch is an
    exact counterfactual rather than a second sample.
    """
    nudges = ep.cycle_nudges
    cleared, return_code, _ = resolve_presentment(
        customer, day, amount, balance_series, nudges=nudges, ambiguity=ambiguity
    )
    if nudges and customer.requires_otp:
        unnudged, _, _ = resolve_presentment(
            customer, day, amount, balance_series, nudges=0, ambiguity=ambiguity
        )
        # The gate opened, and it would not have without the nudges. Attributed
        # whether or not the balance then covered it: the nudge's job is the
        # authentication, and crediting it only when the account also happened
        # to be funded would understate what it did.
        if not unnudged and return_code != rules.RETURN_CODE_AFA_INCOMPLETE:
            ep.otp_completions_from_nudges += 1

    ep.history = ep.history + [
        AgentPaymentHistoryRecord(
            customer_id=customer.customer_id,
            attempt_date=day,
            amount=amount,
            succeeded=cleared,
            return_code=None if cleared else return_code,
        )
    ]
    if not cleared:
        ep.failed_presentments += 1
        if return_code == rules.RETURN_CODE_AFA_INCOMPLETE:
            ep.afa_refused += 1
    return cleared, return_code


def _run_cycle(
    *,
    policy: Policy,
    ep: _Episode,
    customer: Customer,
    agent_customer: AgentCustomer,
    cycle_debit: date,
    cycle_end: date,
    cycle_index: int,
    cycle_attempts: int,
    amount: float,
    balance_series: pd.Series,
    seed: int,
    irritation_direction: IrritationDirection,
    lead_time_days: int = 1,
    notifications_can_cancel: bool = True,
    failure_class: str = rules.FAILURE_CLASS_INSUFFICIENT_FUNDS,
    return_code: str = rules.RETURN_CODE_INSUFFICIENT_FUNDS,
    strict_compliance: bool = True,
    ambiguity: AmbiguityConfig | None = None,
) -> bool:
    """Walk one billing cycle. Returns True if the episode ended inside it.

    ``cycle_attempts`` is the NPCI budget consumed so far IN THIS CYCLE - the
    ceiling is per cycle, so it resets while the episode-level totals on ``ep``
    keep accumulating.
    """
    agent_failure = AgentFailureEvent(
        customer_id=customer.customer_id,
        cycle_start_date=cycle_debit,
        original_debit_date=cycle_debit,
        return_code=return_code,
        failure_class=failure_class,
    )
    cycle_notifications = 0
    ep.cycle_attempts = cycle_attempts
    ep.cycle_notifications = 0
    pending: date | None = None
    day = cycle_debit
    # A zero-lead policy may present on the day it decides, so resolving and
    # deciding can both happen within one day and in either order. This guard
    # stops a misbehaving policy that schedules same-day retries forever from
    # hanging the harness; the attempt ceiling normally ends it first.
    same_day_steps = 0

    while day <= cycle_end:
        resolved_today = False
        if pending == day:
            pending = None
            resolved_today = True
            ep.notifications += 1
            cycle_notifications += 1
            ep.cycle_notifications = cycle_notifications
            # Irritation is measured from THIS cycle's bounce, and from the
            # moment the NOTICE lands rather than the debit: the customer acts
            # on the message, which the issuer sends `lead_time_days` earlier.
            # A longer notice period therefore delivers the message closer to
            # the fresh bounce, which is the mechanism the lead-time ablation
            # prices.
            notice_day = day - timedelta(days=lead_time_days)
            days_since = max(0, (notice_day - cycle_debit).days)
            # notifications_can_cancel=False is the counterfactual world where
            # RBI Sec 6(c) does not exist: the debit is still announced, but the
            # message carries no opt-out. Not legal - it is the control arm for
            # pricing the regulation.
            if notifications_can_cancel and draw_optout(
                customer,
                cycle_notifications,
                days_since,
                optout_rng(customer.customer_id, ep.notifications, seed),
                irritation_direction,
            ):
                # Cancelled on the pre-debit notice, so the debit never ran:
                # no presentment, no bounce.
                ep.outcome, ep.resolved_on = "OPTED_OUT", day
                ep.log.append(
                    _stamped(ep,
                        day, "EXECUTE", day, {}, "opted out on the notice", cycle_index + 1
                    )
                )
                return True

            ep.attempts_used += 1
            cycle_attempts += 1
            ep.cycle_attempts = cycle_attempts
            cleared, code = _record_presentment(
                ep, customer, day, amount, balance_series, ambiguity
            )
            ep.log.append(
                _stamped(ep,
                    day,
                    "EXECUTE",
                    day,
                    {},
                    "debit cleared" if cleared else f"debit failed, code {code}",
                    cycle_index + 1,
                )
            )
            if cleared:
                ep.outcome, ep.recovered_amount, ep.resolved_on = "RECOVERED", amount, day
                return True

        if not resolved_today and pending is None:
            state = PolicyState(
                customer=agent_customer,
                payment_history=tuple(ep.history),
                failure_event=agent_failure,
                today=day,
                attempts_used=cycle_attempts,
                notifications_sent_this_cycle=cycle_notifications,
                nudges_sent_this_cycle=ep.cycle_nudges,
                cycle_end_date=cycle_end,
            )
            decision = policy.decide(state)

            if decision.action == "RETRY":
                assert decision.retry_date is not None  # enforced by the model
                if (
                    not strict_compliance
                    and failure_class in rules.RETRY_IS_BREACH_CLASSES
                ):
                    # Counting mode. A fixed schedule presents against revoked
                    # mandates because it never looks at the return code, and
                    # that is the finding - so the run records the breach
                    # rather than aborting. The acquirer would reject the
                    # presentment, so no attempt is consumed and no money moves.
                    ep.compliance_breaches += 1
                    ep.outcome, ep.resolved_on = "CANCELLED", day
                    ep.log.append(
                        _stamped(ep,
                            day,
                            "BREACH",
                            decision.retry_date,
                            decision.reasoning,
                            f"presented against a {failure_class} mandate",
                            cycle_index + 1,
                        )
                    )
                    return True
                _check_compliance(
                    decision_date=day,
                    retry_date=decision.retry_date,
                    retry_time=decision.retry_time,
                    attempts_used=cycle_attempts,
                    rail=customer.rail,
                    policy_name=policy.name,
                    lead_time_days=lead_time_days,
                    failure_class=failure_class,
                )
                pending = decision.retry_date
                ep.retry_dates.append(decision.retry_date)
                if pending == day:
                    # Zero notice period: the presentment is due today, so it
                    # must resolve in this same tick rather than being stepped
                    # over when the day advances.
                    same_day_steps += 1
                    if same_day_steps > rules.MAX_ATTEMPTS_PER_CYCLE + 2:
                        raise ComplianceViolation(
                            f"{policy.name}: kept scheduling same-day retries on {day}"
                        )
                    continue
                ep.log.append(
                    _stamped(ep,
                        day,
                        "RETRY",
                        decision.retry_date,
                        decision.reasoning,
                        "scheduled",
                        cycle_index + 1,
                        decision.retry_time,
                    )
                )
            elif decision.action == "CANCEL":
                ep.outcome, ep.resolved_on = "CANCELLED", day
                ep.log.append(
                    _stamped(ep,
                        day, "CANCEL", None, decision.reasoning, "terminal", cycle_index + 1
                    )
                )
                return True
            elif decision.action == "HOLD":
                # Not terminal: the mandate is carried into the next cycle,
                # and the carrying cost is charged by the caller.
                ep.log.append(
                    _stamped(ep,
                        day,
                        "HOLD",
                        None,
                        decision.reasoning,
                        "carry to next cycle",
                        cycle_index + 1,
                    )
                )
                return False
            elif decision.action == "NUDGE":
                # OUT OF BAND, and every part of that matters. A nudge is a
                # message the merchant sends directly, not the issuer's
                # pre-debit notice, so it deliberately does NOT touch
                # ``cycle_notifications`` (no RBI Sec 6(c) opt-out opportunity
                # is created and no draw is taken against one) and does NOT
                # touch ``cycle_attempts`` (no NPCI presentment is consumed).
                # It costs goodwill instead, which is the whole trade: a
                # cheaper way to change the customer's state than presenting.
                #
                # Nothing here asks what failure class this is. The nudge
                # raises the AFA-completion probability for whoever needs AFA
                # and is inert for everybody else, which is the arithmetic in
                # generator.afa doing the discriminating rather than a branch.
                ep.nudges += 1
                ep.cycle_nudges += 1
                ep.goodwill_cost += (
                    rules.NUDGE_GOODWILL_COST_FRACTION * agent_customer.annual_ltv
                )
                ep.log.append(
                    _stamped(ep,
                        day,
                        "NUDGE",
                        None,
                        decision.reasoning,
                        f"out of band, nudge {ep.cycle_nudges} this cycle",
                        cycle_index + 1,
                    )
                )
            else:  # WAIT - neither presents nor ends the cycle
                ep.log.append(
                    _stamped(ep,
                        day, decision.action, None, decision.reasoning, "", cycle_index + 1
                    )
                )

        if resolved_today:
            # A presentment resolved and did not end the episode. Consult the
            # policy again before moving on: it may want to act again today.
            same_day_steps += 1
            if same_day_steps <= rules.MAX_ATTEMPTS_PER_CYCLE + 2:
                continue

        same_day_steps = 0
        day += timedelta(days=1)

    return False
